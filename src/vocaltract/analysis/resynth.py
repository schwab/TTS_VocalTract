"""
Level A analysis/resynthesis: recording -> VocalTract frames -> Propeller audio.

No machine learning: classic speech analysis maps a recording onto the 13
VocalTract parameters, and the reference engine renders them. The point is to
measure how much of a real voice survives the 13-byte frame, and to produce
frame sequences that Level B (gradient fitting) can start from.

Mapping (one frame every ``go(14)`` = 9.8 ms; VocalTract interpolates
linearly towards each frame, so frame k carries the features at the *end* of
its interval):

==========  =================================================================
``gp``      Praat f0 -> 100 + 48*log2(f0/110), with a voicing hangover over
            decaying syllable tails; held through each fade-out frame, then
            set to the *next* voiced value so onsets start on pitch
``f1..f4``  Praat Burg formants (voiced frames) -> Hz / 19.53, median-smoothed;
            unvoiced gaps hold the previous shape for their first half and
            take the next syllable's for the second
``ga``      voiced low-band level (calibrated, see below)
``aa``      unvoiced low-band level -> "h" noise through the formants, plus the
            recording's breath share (from HNR) of each voiced frame
``fa,ff``   high-band level where it dominates (s, sh, f...); ff is the
            centroid / 39.06, held constant per consonant (frication is a
            noise-jittered sine that bypasses the resonators)
``vp,vr``   0 (vibrato is already in the f0 track)
``na,nf``   0 (nasals not detected yet)
==========  =================================================================

**Level calibration.** VocalTract's output level for a given ``ga`` depends
strongly on where the formants and pitch are (resonator gains), so amplitudes
can't be set from the recording directly. Instead each source is rendered
alone through the real engine along the tracked formant/pitch trajectory,
its per-frame output level is measured, and the amplitude bytes are solved
(voicing refined twice with the real on/off pattern) so the output level follows the recording. A global gain
puts the 99th-percentile ``ga`` at ``--ga-p99``; if the full render overflows
(VocalTract's 32-bit wrap noise bursts) the gain is backed off and retried.

Usage::

    python -m vocaltract.analysis.resynth assets/recordings/198-209-0000.ogg
    python -m vocaltract.analysis.resynth my_voice.wav --spin   # + Propeller player

Outputs in ``out/resynth/``: ``<name>_resynth.wav``, ``<name>_ab.wav``
(original, then resynthesis), ``<name>_frames.csv``, ``<name>.png``
(spectrograms + parameter lanes) and optionally ``<name>_player.spin``.
"""

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import signal

from ..params import (FORMANT_SCALE, FRICATION_FREQ_SCALE, GP_BASE_HZ, GP_BASE_RAW,
                      OCTAVE_UNIT, PARAM_NAMES)
from ..reference.engine import AA, FA, FF, F1, GA, GP, PARAM_COUNT, SAMPLE_RATE
from ..reference.engine import QueuedFrame, render, stepsize
from .features import Track, _frame_rms, track

HOP_UNITS = 14                       # go(14) per frame
HOP_S = HOP_UNITS * 0.0007           # 9.8 ms
CALIBRATION_LEVEL = 16               # probe amplitude: low enough never to overflow
SILENCE_DB = -45.0                   # frames this far below the loudest are silent...
NOISE_FLOOR_DB = 12.0                # ...or less than this far above the noise floor


@dataclass
class Resynthesis:
    name: str
    frames: np.ndarray               # (N, 13) uint8, frame k targets track time k+1
    audio: np.ndarray                # int32 engine samples at SAMPLE_RATE
    track: Track                     # raw analysis of the recording (before hangover)
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# features -> parameter bytes
# ---------------------------------------------------------------------------

def _fill_gaps(x: np.ndarray, prefer_next: bool = False) -> np.ndarray:
    """Fill NaNs: linear interpolation, or hold the next valid value."""
    x = x.astype(np.float64).copy()
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    idx = np.arange(len(x))
    if prefer_next:
        nxt = np.where(ok, idx, len(x))
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        last = idx[ok][-1]
        nxt[nxt == len(x)] = last
        return x[nxt]
    return np.interp(idx, idx[ok], x[ok])


def pitch_to_gp(f0: np.ndarray) -> np.ndarray:
    return GP_BASE_RAW + OCTAVE_UNIT * np.log2(f0 / GP_BASE_HZ)


def gp_to_pitch(gp: np.ndarray) -> np.ndarray:
    return GP_BASE_HZ * 2.0 ** ((np.asarray(gp, dtype=np.float64) - GP_BASE_RAW) / OCTAVE_UNIT)


def _to_byte(x: np.ndarray, lo: int = 0) -> np.ndarray:
    return np.clip(np.round(np.nan_to_num(x)), lo, 255).astype(np.int64)


def fricative_weight(tr: Track) -> np.ndarray:
    """0 for vowels .. 1 for s/sh: how much a frame's energy is high-band noise."""
    total = np.sqrt(tr.low_rms ** 2 + tr.high_rms ** 2)
    hf_ratio = tr.high_rms / np.maximum(total, 1e-12)
    return np.clip((hf_ratio - 0.35) / 0.3, 0.0, 1.0)


def extend_voicing(tr: Track, max_frames: int = 5, drop_db: float = 15.0) -> Track:
    """Voicing hangover: carry voicing past Praat's offset while the voice decays.

    Praat's voicing decision ends a syllable while its tail is still only a
    few dB down (on the LibriSpeech samples: ~3 dB one frame later, ~6-7 dB
    two frames later). Cutting ``ga`` there and handing the tail to aspiration
    noise is what made syllable endings sound garbled. Unvoiced frames right
    after an offset are relabelled voiced (pitch held) while the low band
    stays within ``drop_db`` of the last voiced frame and the frame isn't
    frication-dominated, for at most ``max_frames``.
    """
    f0 = tr.f0.copy()
    frication = fricative_weight(tr) > 0
    voiced = ~np.isnan(tr.f0)
    for k in np.flatnonzero(voiced[:-1] & ~voiced[1:]):
        floor = tr.low_rms[k] * 10 ** (-drop_db / 20)
        for j in range(k + 1, min(k + 1 + max_frames, len(f0))):
            # A decaying vowel loses its low band first, so its high/low ratio
            # rises; only stop for frication if the high band itself grew.
            fricative_onset = frication[j] and tr.high_rms[j] > 2 * tr.high_rms[k]
            if voiced[j] or fricative_onset or tr.low_rms[j] < floor:
                break
            f0[j] = f0[k]
    return replace(tr, f0=f0)


def _bridge_gaps(x: np.ndarray, voiced: np.ndarray, hold) -> np.ndarray:
    """Fill each unvoiced run: the first ``hold(run_length)`` frames keep the
    previous voiced value, the rest take the next voiced value.

    VocalTract glides linearly into every frame, so what matters is which
    *intervals* change: pitch must not move during the fade-out interval
    (voiced -> unvoiced, ``hold=1``), and formants should hold through the
    decaying tail and move to the next syllable's shape before its onset.
    """
    x = x.astype(np.float64).copy()
    n = len(x)
    if not voiced.any():
        return _fill_gaps(x)
    k = 0
    while k < n:
        if voiced[k]:
            k += 1
            continue
        e = k
        while e + 1 < n and not voiced[e + 1]:
            e += 1
        prev = x[k - 1] if k > 0 else x[e + 1]
        nxt = x[e + 1] if e + 1 < n else prev
        h = min(hold(e - k + 1), e - k + 1) if k > 0 else 0
        x[k:k + h] = prev
        x[k + h:e + 1] = nxt
        k = e + 1
    return x


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of the True runs in ``mask``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2])]


def _smooth_runs(x: np.ndarray, mask: np.ndarray, width: int) -> np.ndarray:
    """Hann-window moving average inside each run of ``mask`` (edges renormalized)."""
    if width < 3:
        return x
    out = x.copy()
    for s, e in _runs(mask):
        seg = x[s:e + 1]
        n = min(width, len(seg) | 1 if len(seg) % 2 else len(seg) - 1)   # odd, <= run length
        if n < 3:
            continue
        w = np.hanning(n + 2)[1:-1]
        out[s:e + 1] = np.convolve(seg, w, "same") / np.convolve(np.ones_like(seg), w, "same")
    return out


def _voiced_track(x: np.ndarray, voiced: np.ndarray, kernel: int, smooth: int = 0) -> np.ndarray:
    """Values on voiced frames only: gap-filled within voicing, median-smoothed,
    then optionally Hann-smoothed over ``smooth`` frames within each voiced run."""
    out = np.full(len(x), np.nan)
    if voiced.sum() == 0:
        return out
    v = _fill_gaps(np.where(voiced, x, np.nan))
    v[voiced] = signal.medfilt(v[voiced], kernel) if voiced.sum() >= kernel else v[voiced]
    v = _smooth_runs(v, voiced, smooth)
    out[voiced] = v[voiced]
    return out


def shape_frames(tr: Track, formant_smooth: int = 0) -> np.ndarray:
    """Pitch, formant and frication-frequency bytes (amplitudes left at 0)."""
    n = len(tr.times) - 1
    frames = np.zeros((n, PARAM_COUNT), dtype=np.int64)
    voiced = tr.voiced[1:]

    gp = _voiced_track(pitch_to_gp(tr.f0[1:]), voiced, 3)
    frames[:, GP] = _to_byte(_bridge_gaps(gp, voiced, hold=lambda length: 1))

    if voiced.any():
        fmt = np.stack([_bridge_gaps(_voiced_track(tr.formants[1:, i], voiced, 5, formant_smooth),
                                     voiced,
                                     hold=lambda length: (length + 1) // 2)
                        for i in range(4)], 1)
    else:                                   # whisper: no voicing to anchor on
        fmt = np.stack([signal.medfilt(_fill_gaps(tr.formants[1:, i]), 5) for i in range(4)], 1)
    fmt = np.sort(np.nan_to_num(fmt, nan=3500.0), axis=1)
    frames[:, F1:F1 + 4] = _to_byte(fmt / FORMANT_SCALE, lo=1)

    frames[:, FF] = _to_byte(segment_ff(tr) / FRICATION_FREQ_SCALE, lo=1)
    return frames


def segment_ff(tr: Track) -> np.ndarray:
    """Frication frequency, held constant across each frication segment.

    VocalTract's frication is a noise-jittered *sine* at ff, not white noise,
    so a per-frame centroid makes the hiss audibly change pitch inside every
    s/sh/f -- a warbling, robotic consonant. Like Chip's hand-made 'seven'
    (one ff per consonant), each run of fricative frames gets the median
    centroid of the run; frames outside runs take the nearest run's value.
    """
    centroid = _fill_gaps(tr.high_centroid[1:])
    centroid = np.nan_to_num(centroid, nan=4000.0)
    fric = fricative_weight(tr)[1:] > 0
    ff = np.full(len(centroid), np.nan)
    k = 0
    while k < len(fric):
        if not fric[k]:
            k += 1
            continue
        e = k
        while e + 1 < len(fric) and fric[e + 1]:
            e += 1
        ff[k:e + 1] = np.median(centroid[k:e + 1])
        k = e + 1
    return _fill_gaps(ff) if fric.any() else centroid


def source_targets(tr: Track, breathiness: float = 0.0) -> dict[int, np.ndarray]:
    """Desired output RMS per frame for each source (recording units)."""
    voiced = tr.voiced[1:]
    low, high = tr.low_rms[1:], tr.high_rms[1:]
    total = np.sqrt(low ** 2 + high ** 2)
    floor = np.percentile(total, 10)                           # room noise between words
    audible = total > max(total.max() * 10 ** (SILENCE_DB / 20),
                          floor * 10 ** (NOISE_FLOOR_DB / 20))
    fric_weight = fricative_weight(tr)[1:]
    # Breathiness: the recording's noise share of each voiced frame (from
    # Praat's HNR) goes to aspiration, which VocalTract sends through the same
    # formants. Breath noise excites the formants between harmonics, so vowels
    # stay identifiable at high f0, where the narrow resonators would otherwise
    # land between the widely spaced harmonics.
    noise_share = np.zeros(len(low))
    if breathiness > 0:
        hnr = np.nan_to_num(tr.hnr[1:], nan=20.0)
        noise_share = np.clip(breathiness / (1 + 10 ** (hnr / 10)), 0.0, 0.9)
    return {
        GA: np.where(voiced & audible, low * np.sqrt(1 - noise_share), 0.0),
        AA: np.where(audible, np.where(voiced, low * np.sqrt(noise_share), low), 0.0),
        FA: np.where(audible, high * fric_weight, 0.0),
    }


# ---------------------------------------------------------------------------
# rendering and calibration
# ---------------------------------------------------------------------------

def render_frames(frames: np.ndarray, tail_seconds: float = 0.1) -> np.ndarray:
    step = stepsize(HOP_UNITS)
    queued = [QueuedFrame(0, tuple(int(v) for v in row), step) for row in frames]
    return render(queued, n_tracts=1, tail_seconds=tail_seconds)[0]


def _measure(audio: np.ndarray, n_frames: int) -> np.ndarray:
    """Output RMS at each frame's target instant (end of its interval)."""
    times = (np.arange(n_frames) + 1) * HOP_S
    return _frame_rms(audio.astype(np.float64), SAMPLE_RATE, times, 2 * HOP_S)


def _response(shape: np.ndarray, param: int, mask: np.ndarray, level) -> np.ndarray:
    frames = shape.copy()
    frames[:, param] = np.where(mask, level, 0)
    return _measure(render_frames(frames), len(frames))


def calibrate(shape: np.ndarray, targets: dict[int, np.ndarray], ga_p99: float,
              refine: int = 2) -> tuple[np.ndarray, float]:
    """Solve amplitude bytes so each source's output RMS follows its target.

    The probe renders play each source *continuously* along the trajectory, so
    the measured gain is the resonators' steady-state gain at that frame. (A
    masked probe would under-measure isolated frames -- VocalTract interpolates
    towards each frame, so a 1-frame burst never reaches its level -- and the
    solver would then overshoot them into clicks.)
    """
    gains = {}
    per_byte = {}
    always = np.ones(len(shape), dtype=bool)
    for param, target in targets.items():
        if (target > 0).any():
            per_byte[param] = _response(shape, param, always, CALIBRATION_LEVEL) / CALIBRATION_LEVEL

    voiced = targets[GA] > 0
    raw_ga = targets[GA][voiced] / np.maximum(per_byte[GA][voiced], 1e-9)
    scale = ga_p99 / np.percentile(raw_ga, 99)            # engine units per recording unit

    for param, target in targets.items():
        mask = target > 0
        amp = np.zeros(len(target))
        if param in per_byte:
            amp[mask] = scale * target[mask] / np.maximum(per_byte[param][mask], 1e-9)
            if param != GA:                     # noise never louder than the voice
                amp = np.minimum(amp, ga_p99)
            for _ in range(refine if param == GA else 0):
                got = _response(shape, param, mask, _to_byte(amp))
                ratio = scale * target[mask] / np.maximum(got[mask], 1e-9)
                amp[mask] *= np.clip(ratio, 0.5, 2.0) ** 0.8
        gains[param] = _to_byte(amp)

    frames = shape.copy()
    for param, amp in gains.items():
        frames[:, param] = amp
    return frames, scale


def drop_islands(frames: np.ndarray) -> np.ndarray:
    """Silence runs of sound that have neither voicing nor frication.

    Between words these are breaths, lip noise and room noise that passed the
    gate. Real consonants carry frication, and in-word aspiration ('h', stop
    releases into a vowel) is contiguous with voicing, so both survive. Sent
    through VocalTract's narrow resonators, a soft broadband breath turns into
    a tonal chirp, which is why these stood out in the silences.
    """
    frames = frames.copy()
    sounding = frames[:, [GA, AA, FA]].any(axis=1)
    for s, e in _runs(sounding):
        if not frames[s:e + 1, GA].any() and not frames[s:e + 1, FA].any():
            frames[s:e + 1, [GA, AA, FA]] = 0
    return frames


def _overflowed(frames: np.ndarray, audio: np.ndarray) -> bool:
    """Compare the full render with the sum of its sources rendered alone.

    The engine is linear apart from 32-bit wraparound, so a large mismatch
    means VocalTract overflowed (the 'high-amplitude noise bursts' its
    documentation warns about).
    """
    parts = np.zeros(len(audio))
    for param in (GA, AA, FA):
        solo = frames.copy()
        for other in (GA, AA, FA):
            if other != param:
                solo[:, other] = 0
        parts += render_frames(solo).astype(np.float64)[:len(audio)]
    err = np.abs(parts - audio).max()
    return err > 0.05 * max(np.abs(audio).max(), 1)


def resynthesize(audio: np.ndarray, sr: int, name: str = "clip", *, ga_p99: float = 110.0,
                 max_formant: float | None = None, f0_min: float = 60.0,
                 f0_max: float = 500.0, hangover: int = 5, formant_smooth: int = 0,
                 keep_breaths: bool = False, breathiness: float = 1.0) -> Resynthesis:
    raw = track(audio, sr, HOP_S, f0_min=f0_min, f0_max=f0_max, max_formant=max_formant)
    tr = extend_voicing(raw, max_frames=hangover) if hangover else raw
    shape = shape_frames(tr, formant_smooth)
    targets = source_targets(tr, breathiness)
    for attempt in range(5):
        frames, _ = calibrate(shape, targets, ga_p99)
        if not keep_breaths:
            frames = drop_islands(frames)
        out = render_frames(frames)
        if not _overflowed(frames, out):
            break
        ga_p99 *= 0.7
    res = Resynthesis(name, frames.astype(np.uint8), out, raw)   # evaluate against raw Praat
    res.metrics["ga_p99"] = ga_p99
    res.metrics["overflow_retries"] = attempt
    return res


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def edge_metrics(frames: np.ndarray) -> dict:
    """How syllable endings are rendered (where 'garbled tails' come from).

    At each voicing offset (ga > 0 -> ga == 0):
    - ``offset_pitch_swing``: fraction where gp moves >= 6 units (1.5
      semitones) during the fade-out interval -- an audible pitch swoop
    - ``offset_cut_db``: median level of the last voiced ga relative to its
      syllable's median ga; near 0 dB means the voice is chopped off at full
      level instead of decaying
    """
    ga = frames[:, GA].astype(np.int64)
    gp = frames[:, GP].astype(np.int64)
    offsets = np.flatnonzero((ga[:-1] > 0) & (ga[1:] == 0))
    if not offsets.size:
        return {"offset_pitch_swing": float("nan"), "offset_cut_db": float("nan")}
    swing = np.abs(gp[offsets + 1] - gp[offsets]) >= 6
    cut = []
    for k in offsets:
        j = k
        while j > 0 and ga[j - 1] > 0:
            j -= 1
        cut.append(20 * np.log10(ga[k] / max(np.median(ga[j:k + 1]), 1)))
    return {"offset_pitch_swing": float(swing.mean()), "offset_cut_db": float(np.median(cut))}


def _mel_db(x: np.ndarray, sr: int, n_mels: int = 40, nfft: int = 512) -> np.ndarray:
    hop = int(sr * 0.01)
    _, _, z = signal.stft(x, sr, nperseg=nfft, noverlap=nfft - hop)
    p = np.abs(z) ** 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(60), mel(sr / 2), n_mels + 2))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return 10 * np.log10(fb @ p + 1e-12)


def evaluate(res: Resynthesis, audio: np.ndarray, sr: int, f0_min=60.0, f0_max=500.0) -> dict:
    """Compare the resynthesis with the original.

    Pitch and voicing are re-tracked on the resynthesis; formants compare the
    frames with the original track (see note below); log-mel distance is the
    gain-aligned mean |dB| difference over audible frames.
    """
    synth = signal.resample_poly(res.audio.astype(np.float64) / 2**31, sr, SAMPLE_RATE)
    n = min(len(synth), len(audio))
    synth, orig = synth[:n], np.asarray(audio, dtype=np.float64)[:n]
    tr_o = res.track
    tr_s = track(synth, sr, HOP_S, f0_min=f0_min, f0_max=f0_max, max_formant=tr_o.max_formant)
    m = min(len(tr_o.times), len(tr_s.times))

    both = tr_o.voiced[:m] & tr_s.voiced[:m]
    cents = 1200 * np.abs(np.log2(tr_s.f0[:m][both] / tr_o.f0[:m][both]))
    voicing_agree = np.mean(tr_o.voiced[:m] == tr_s.voiced[:m])

    # Formants: compare what the frames *ask for* with the original track
    # (byte quantization + smoothing). Re-tracking formants on the engine's
    # output is NOT a fair test: on static renders whose spectral peaks are
    # verifiably in place, Praat splits F2 and shifts F3/F4 down a slot,
    # because VocalTract's glottal pulse adds spectral bumps of its own.
    told = res.frames[:, F1:F1 + 4].astype(np.float64) * FORMANT_SCALE   # frame k <-> time k+1
    voiced = res.frames[:, GA] > 0
    fmt_err = []
    for i in range(4):
        orig_f = tr_o.formants[1:len(told) + 1, i]
        ok = voiced[:len(orig_f)] & ~np.isnan(orig_f)
        fmt_err.append(float(np.median(np.abs(told[:len(orig_f), i][ok] - orig_f[ok]))))

    mo, ms = _mel_db(orig, sr), _mel_db(synth, sr)
    k = min(mo.shape[1], ms.shape[1])
    mo, ms = mo[:, :k], ms[:, :k]
    loud = mo.mean(0) > mo.mean(0).max() - 40
    offset = np.median((ms - mo)[:, loud])                # ignore overall gain
    ms = ms - offset
    floor = mo.max() - 60                                  # don't score silence vs room noise
    lsd = float(np.mean(np.abs(np.maximum(ms, floor) - np.maximum(mo, floor))[:, loud]))

    try:
        from pystoi import stoi
        intelligibility = float(stoi(orig, synth, sr, extended=False))
    except ImportError:
        intelligibility = float("nan")

    return {
        **edge_metrics(res.frames),
        "stoi": intelligibility,
        "voiced_fraction": float(tr_o.voiced.mean()),
        "voicing_agreement": float(voicing_agree),
        "pitch_error_cents_median": float(np.median(cents)) if cents.size else float("nan"),
        "formant_error_hz_median": fmt_err,
        "log_mel_distance_db": lsd,
    }


def asr_metrics(res: Resynthesis, audio: np.ndarray, sr: int, model: str,
                reference: str | None = None) -> dict:
    """Whisper word error rate for the original recording and the resynthesis.

    Without a reference transcript, the original's own transcript is used as
    the reference (so the original scores 0% by construction).
    """
    from .intelligibility import transcribe, word_error_rate

    hyp_orig = transcribe(audio, sr, model)
    hyp_synth = transcribe(res.audio.astype(np.float64), SAMPLE_RATE, model)
    ref = reference if reference else hyp_orig
    orig = word_error_rate(ref, hyp_orig)
    synth = word_error_rate(ref, hyp_synth)
    return {
        "asr_model": model,
        "asr_reference": "transcript file" if reference else "ASR of original",
        "wer_original": orig.wer,
        "wer_resynth": synth.wer,
        "wer_resynth_detail": synth.summary(),
        "asr_text_original": hyp_orig,
        "asr_text_resynth": hyp_synth,
    }


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def write_outputs(res: Resynthesis, audio: np.ndarray, sr: int, out_dir: Path,
                  spin: bool = False) -> list[Path]:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    synth = res.audio.astype(np.float64) / 2**31
    synth *= 0.9 / max(np.abs(synth).max(), 1e-12)

    p = out_dir / f"{res.name}_resynth.wav"
    sf.write(p, synth, SAMPLE_RATE, subtype="PCM_16")
    paths.append(p)

    orig = signal.resample_poly(np.asarray(audio, dtype=np.float64), SAMPLE_RATE, sr)
    orig *= 0.9 / max(np.abs(orig).max(), 1e-12)
    gap = np.zeros(int(0.75 * SAMPLE_RATE))
    p = out_dir / f"{res.name}_ab.wav"
    sf.write(p, np.concatenate([orig, gap, synth]), SAMPLE_RATE, subtype="PCM_16")
    paths.append(p)

    p = out_dir / f"{res.name}_frames.csv"
    header = "time_s," + ",".join(PARAM_NAMES)
    times = (np.arange(len(res.frames)) + 1) * HOP_S
    np.savetxt(p, np.column_stack([times, res.frames]), delimiter=",", header=header,
               comments="", fmt=["%.4f"] + ["%d"] * PARAM_COUNT)
    paths.append(p)

    if "asr_text_resynth" in res.metrics:
        m = res.metrics
        p = out_dir / f"{res.name}_asr.txt"
        p.write_text(f"model: {m['asr_model']}  reference: {m['asr_reference']}\n"
                     f"original    (WER {m['wer_original']:.1%}): {m['asr_text_original']}\n"
                     f"resynthesis (WER {m['wer_resynth']:.1%}): {m['asr_text_resynth']}\n",
                     encoding="utf-8")
        paths.append(p)

    paths.append(plot(res, audio, sr, out_dir / f"{res.name}.png"))
    if spin:
        paths.append(export_spin(res.frames, out_dir / f"{res.name}_player.spin", res.name))
    return paths


def plot(res: Resynthesis, audio: np.ndarray, sr: int, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 3, 2]})
    synth = res.audio.astype(np.float64) / 2**31
    for a, (x, rate, title) in zip(ax[:2], [(audio, sr, "original"),
                                            (synth, SAMPLE_RATE, "VocalTract resynthesis")]):
        a.specgram(x, NFFT=512, Fs=rate, noverlap=384, cmap="magma",
                   vmin=None if title == "original" else None)
        a.set_ylim(0, min(sr, SAMPLE_RATE) / 2)
        a.set_ylabel("Hz")
        a.set_title(title)
    t = (np.arange(len(res.frames)) + 1) * HOP_S
    on = res.frames[:, GA] > 0
    for i in range(4):
        f = res.frames[:, F1 + i].astype(float) * FORMANT_SCALE
        ax[0].plot(t, np.where(on, f, np.nan), ".", ms=1.5, color="cyan")
    ax[0].plot(t, np.where(on, gp_to_pitch(res.frames[:, GP]), np.nan), ".", ms=1.5, color="lime")
    for name, idx in (("ga", GA), ("aa", AA), ("fa", FA)):
        ax[2].plot(t, res.frames[:, idx], label=name, lw=0.8)
    ax[2].set_ylabel("byte")
    ax[2].set_xlabel("s")
    ax[2].legend(loc="upper right")
    ax[2].set_title("source amplitudes (formants cyan / pitch green on the original)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def export_spin(frames: np.ndarray, path: Path, name: str) -> Path:
    """Write a minimal Propeller program that plays the frames on real hardware.

    Uses VocalTract's delta-modulation output on P10 (+) / P11 (-), the
    Propeller Demo Board's audio pins. Needs VocalTract.spin alongside.
    """
    size = frames.size
    if size > 24_000:
        print(f"warning: {size} frame bytes may not fit in 32 KB hub RAM; trim the clip")
    rows = "\n".join("        byte " + ",".join(str(int(v)) for v in row) for row in frames)
    path.write_text(f"""{{{{ {name}: VocalTract frames from vocaltract.analysis.resynth (level A).
   {len(frames)} frames, go({HOP_UNITS}) each = {len(frames) * HOP_S:.1f} s.
   Audio out on P10/P11 (Propeller Demo Board). Needs VocalTract.spin. }}}}

CON

  _clkmode = xtal1 + pll16x
  _xinfreq = 5_000_000

  frame_count = {len(frames)}

OBJ

  v : "VocalTract"

VAR

  byte  p[13]

PUB start | i

  v.start(@p, 10, 11, -1)
  repeat i from 0 to frame_count - 1
    bytemove(@p, @frames + i * 13, 13)
    v.go({HOP_UNITS})
  repeat

DAT

frames
{rows}
""", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Level A: recording -> VocalTract frames -> audio")
    ap.add_argument("inputs", nargs="+", type=Path, help="audio files (wav/ogg/flac)")
    ap.add_argument("--out-dir", type=Path, default=Path("out") / "resynth")
    ap.add_argument("--start", type=float, default=0.0, help="trim: start time (s)")
    ap.add_argument("--duration", type=float, help="trim: length (s)")
    ap.add_argument("--max-formant", type=float, help="Praat formant ceiling (default by voice)")
    ap.add_argument("--f0-min", type=float, default=60.0)
    ap.add_argument("--f0-max", type=float, default=500.0)
    ap.add_argument("--ga-p99", type=float, default=110.0,
                    help="99th-percentile glottal amplitude byte (loudness vs overflow)")
    ap.add_argument("--hangover", type=int, default=5,
                    help="max frames of voicing carried over decaying syllable tails (0 = off)")
    ap.add_argument("--breathiness", type=float, default=1.0,
                    help="aspiration mixed into voiced frames, x the recording's noise share "
                         "(from HNR); 0 = off, 1 = matched (default; cut female WER 53%%->43%%)")
    ap.add_argument("--formant-smooth", type=int, default=0,
                    help="Hann smoothing of formant tracks within voiced runs, in frames "
                         "(0 = off; raised WER on both test voices -- fast formant motion "
                         "carries consonant cues)")
    ap.add_argument("--keep-breaths", action="store_true",
                    help="keep sound islands with no voicing or frication (breaths, lip noise)")
    ap.add_argument("--asr", default="small.en",
                    help="Whisper model for the word-error-rate metric, or 'none' (default small.en)")
    ap.add_argument("--spin", action="store_true", help="also write a Propeller player .spin")
    args = ap.parse_args(argv)

    import soundfile as sf

    for path in args.inputs:
        audio, sr = sf.read(path, always_2d=True)
        audio = audio.mean(axis=1)
        lo = int(args.start * sr)
        hi = len(audio) if args.duration is None else lo + int(args.duration * sr)
        audio = audio[lo:hi]

        res = resynthesize(audio, sr, path.stem, ga_p99=args.ga_p99,
                           max_formant=args.max_formant, f0_min=args.f0_min, f0_max=args.f0_max,
                           hangover=args.hangover, breathiness=args.breathiness,
                           formant_smooth=args.formant_smooth, keep_breaths=args.keep_breaths)
        res.metrics.update(evaluate(res, audio, sr, args.f0_min, args.f0_max))
        if args.asr != "none":
            from .intelligibility import reference_for
            trimmed = args.start > 0 or args.duration is not None
            ref = None if trimmed else reference_for(path)   # transcript covers the whole file
            res.metrics.update(asr_metrics(res, audio, sr, args.asr, ref))
        paths = write_outputs(res, audio, sr, args.out_dir, spin=args.spin)

        m = res.metrics
        print(f"{path.name}: {len(res.frames)} frames, {len(audio) / sr:.1f} s, "
              f"formant ceiling {res.track.max_formant:.0f} Hz")
        print(f"  voiced {m['voiced_fraction']:.0%}, voicing agreement {m['voicing_agreement']:.0%}, "
              f"pitch error {m['pitch_error_cents_median']:.0f} cents (median)")
        print("  formant frames vs track (median Hz): "
              + ", ".join(f"F{i + 1} {e:.0f}" for i, e in enumerate(m["formant_error_hz_median"])))
        print(f"  syllable endings: pitch swoop in {m['offset_pitch_swing']:.0%}, "
              f"voice cut at {m['offset_cut_db']:+.1f} dB re syllable")
        print(f"  log-mel distance {m['log_mel_distance_db']:.1f} dB, STOI {m['stoi']:.2f}, "
              f"ga p99 {m['ga_p99']:.0f}, overflow retries {m['overflow_retries']}")
        if "wer_resynth" in m:
            print(f"  Whisper {m['asr_model']} WER: original {m['wer_original']:.0%}, "
                  f"resynthesis {m['wer_resynth_detail']}  [reference: {m['asr_reference']}]")
            print(f"    heard: {m['asr_text_resynth']}")
        for p in paths:
            print(f"  -> {p}")


if __name__ == "__main__":
    main()
