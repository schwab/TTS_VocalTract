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
``gp``      Praat f0 -> 100 + 48*log2(f0/110); held across unvoiced gaps at the
            *next* voiced value so onsets start on pitch
``f1..f4``  Praat Burg formants -> Hz / 19.53, gap-filled and median-smoothed
``ga``      voiced low-band level (calibrated, see below)
``aa``      unvoiced low-band level -> breathy / "h" noise through the formants
``fa,ff``   high-band level where it dominates (s, sh, f...) and its centroid
            / 39.06 -- frication bypasses the resonators in VocalTract
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
from dataclasses import dataclass, field
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
    track: Track
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


def shape_frames(tr: Track) -> np.ndarray:
    """Pitch, formant and frication-frequency bytes (amplitudes left at 0)."""
    n = len(tr.times) - 1
    frames = np.zeros((n, PARAM_COUNT), dtype=np.int64)

    gp = pitch_to_gp(tr.f0[1:])
    gp[~np.isnan(gp)] = signal.medfilt(gp[~np.isnan(gp)], 3) if (~np.isnan(gp)).sum() >= 3 \
        else gp[~np.isnan(gp)]
    frames[:, GP] = _to_byte(_fill_gaps(gp, prefer_next=True))

    fmt = np.stack([signal.medfilt(_fill_gaps(tr.formants[1:, i]), 5) for i in range(4)], 1)
    fmt = np.sort(np.nan_to_num(fmt, nan=3500.0), axis=1)
    frames[:, F1:F1 + 4] = _to_byte(fmt / FORMANT_SCALE, lo=1)

    ff = signal.medfilt(_fill_gaps(tr.high_centroid[1:]), 5)
    frames[:, FF] = _to_byte(np.nan_to_num(ff, nan=4000.0) / FRICATION_FREQ_SCALE, lo=1)
    return frames


def source_targets(tr: Track) -> dict[int, np.ndarray]:
    """Desired output RMS per frame for each source (recording units)."""
    voiced = tr.voiced[1:]
    low, high = tr.low_rms[1:], tr.high_rms[1:]
    total = np.sqrt(low ** 2 + high ** 2)
    floor = np.percentile(total, 10)                           # room noise between words
    audible = total > max(total.max() * 10 ** (SILENCE_DB / 20),
                          floor * 10 ** (NOISE_FLOOR_DB / 20))

    hf_ratio = high / np.maximum(total, 1e-12)
    fric_weight = np.clip((hf_ratio - 0.35) / 0.3, 0.0, 1.0)   # 0 for vowels, 1 for s/sh
    return {
        GA: np.where(voiced & audible, low, 0.0),
        AA: np.where(~voiced & audible, low, 0.0),
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
                 f0_max: float = 500.0) -> Resynthesis:
    tr = track(audio, sr, HOP_S, f0_min=f0_min, f0_max=f0_max, max_formant=max_formant)
    shape = shape_frames(tr)
    targets = source_targets(tr)
    for attempt in range(5):
        frames, _ = calibrate(shape, targets, ga_p99)
        out = render_frames(frames)
        if not _overflowed(frames, out):
            break
        ga_p99 *= 0.7
    res = Resynthesis(name, frames.astype(np.uint8), out, tr)
    res.metrics["ga_p99"] = ga_p99
    res.metrics["overflow_retries"] = attempt
    return res


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

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

    return {
        "voiced_fraction": float(tr_o.voiced.mean()),
        "voicing_agreement": float(voicing_agree),
        "pitch_error_cents_median": float(np.median(cents)) if cents.size else float("nan"),
        "formant_error_hz_median": fmt_err,
        "log_mel_distance_db": lsd,
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
                           max_formant=args.max_formant, f0_min=args.f0_min, f0_max=args.f0_max)
        res.metrics.update(evaluate(res, audio, sr, args.f0_min, args.f0_max))
        paths = write_outputs(res, audio, sr, args.out_dir, spin=args.spin)

        m = res.metrics
        print(f"{path.name}: {len(res.frames)} frames, {len(audio) / sr:.1f} s, "
              f"formant ceiling {res.track.max_formant:.0f} Hz")
        print(f"  voiced {m['voiced_fraction']:.0%}, voicing agreement {m['voicing_agreement']:.0%}, "
              f"pitch error {m['pitch_error_cents_median']:.0f} cents (median)")
        print("  formant frames vs track (median Hz): "
              + ", ".join(f"F{i + 1} {e:.0f}" for i, e in enumerate(m["formant_error_hz_median"])))
        print(f"  log-mel distance {m['log_mel_distance_db']:.1f} dB, "
              f"ga p99 {m['ga_p99']:.0f}, overflow retries {m['overflow_retries']}")
        for p in paths:
            print(f"  -> {p}")


if __name__ == "__main__":
    main()
