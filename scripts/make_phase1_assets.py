"""
Build the Phase 1 Step 7 overfit targets in assets/phase1/.

Every target is exactly 1.0 s of 16 kHz mono audio (16000 samples), matching
the Phase 1 pipeline (sample_rate=16000, hop=320 -> 50 frames). Each WAV has a
sibling .npz with parameters on that same 50-frame grid, normalized the way
``vocaltract.params`` expects (raw byte / 255), sampled at frame centres
(t + 0.5) * 20 ms -- the convention of ``source.upsample_frames``
(F.interpolate, align_corners=False).

Targets, easiest first:

  synthetic_vowel_ah   classic source-filter /a/ (impulse train -> glottal tilt
                       -> 4 resonators with natural bandwidths); NOT VocalTract.
                       Params: ground truth in physical units.
  vt_vowel_ah          VocalTract reference engine, steady /a/ at 120 Hz.
  vt_glide_ah_ee       VocalTract, /a/ -> /i/ glide with a 110 -> 150 Hz rise.
  vt_sung_oo_vibrato   VocalTract, sung /u/ at 196 Hz with 5.3 Hz vibrato.
  vt_seven             VocalTract, one "seven" from SingingDemoSeven (tenor):
                       frication + nasal, which the Phase 1 source can't make.
  speech_male          1 s of LibriSpeech 5703-47212-0000 ("during the summer
                       months"); params from level A analysis (approximate).
  speech_female        1 s of LibriSpeech 198-209-0000 ("the next morning");
                       params from level A analysis (approximate).

Run from the repo root (needs the analysis extras + openai-whisper):

    python scripts/make_phase1_assets.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from vocaltract.params import PARAM_NAMES
from vocaltract.reference.engine import (AA, F1, GA, GP, SAMPLE_RATE as VT_RATE, VP, VR,
                                         QueuedFrame, render, stepsize)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "phase1"
RATE = 16_000
HOP = 320
N_FRAMES = 50
N_SAMPLES = N_FRAMES * HOP
PEAK = 0.5                       # -6 dBFS: headroom for the pipeline's output
TIME_UNIT_S = 0.0007             # VocalTract go() unit at pace 100

# Peterson & Barney (1952) adult male averages; F4 nominal
VOWELS = {
    "ah": (730, 1090, 2440, 3400),
    "ee": (270, 2290, 3010, 3600),
    "oo": (300, 870, 2240, 3300),
}

FRAME_CENTRES_S = (np.arange(N_FRAMES) + 0.5) * HOP / RATE


def hz_to_gp(hz: float) -> float:
    return 100 + 48 * np.log2(hz / 110.0)


def formant_bytes(hz: tuple) -> list[int]:
    return [int(round(f / 19.53)) for f in hz]


def finish(audio: np.ndarray, rate: int) -> np.ndarray:
    """Resample to 16 kHz, cut/pad to exactly 1 s, normalize to PEAK."""
    x = np.asarray(audio, dtype=np.float64)
    if rate != RATE:
        x = signal.resample_poly(x, RATE, rate)
    x = np.pad(x, (0, max(0, N_SAMPLES - len(x))))[:N_SAMPLES]
    return x * (PEAK / np.abs(x).max())


# ---------------------------------------------------------------------------
# VocalTract targets: frame list -> render + exact parameter trajectory
# ---------------------------------------------------------------------------

def vt_target(frames: list[tuple[list[int], int]], pace: int = 100):
    """Render (params13, time) frames; return audio and the 50-frame trajectory.

    VocalTract interpolates every parameter linearly from its previous value
    to each frame's value over the frame's duration, starting from all zeros;
    the trajectory is that piecewise-linear curve sampled at frame centres.
    """
    queued = [QueuedFrame(0, tuple(int(v) for v in p), stepsize(t, pace)) for p, t in frames]
    audio = render(queued, n_tracts=1, tail_seconds=0.0)[0].astype(np.float64) / 2**31

    knots_t, knots_v, t = [0.0], [np.zeros(13)], 0.0
    for p, time in frames:
        t += max(time * 100 // pace, 2) * TIME_UNIT_S
        knots_t.append(t)
        knots_v.append(np.asarray(p, dtype=np.float64))
    knots_v = np.stack(knots_v)
    raw = np.stack([np.interp(FRAME_CENTRES_S, knots_t, knots_v[:, i]) for i in range(13)], 1)
    return audio, raw


def vowel_frame(hz_formants, f0, ga=60, vp=0, vr=0):
    p = [0] * 13
    p[GA], p[GP], p[VP], p[VR] = ga, int(round(hz_to_gp(f0))), vp, vr
    p[F1:F1 + 4] = formant_bytes(hz_formants)
    return p


def make_vt_vowel():
    p = vowel_frame(VOWELS["ah"], 120)
    return vt_target([(p, 2), (p, 1500)])


def make_vt_glide():
    a = vowel_frame(VOWELS["ah"], 110)
    e = vowel_frame(VOWELS["ee"], 150)
    # 250 ms /a/, 500 ms glide, rest /i/  (time units of 0.7 ms)
    return vt_target([(a, 2), (a, 357), (e, 714), (e, 700)])


def make_vt_sung():
    # G3 (196 Hz), vibrato vr=70 -> 5.3 Hz, vp=6 -> +/-1/16 octave (+/-0.75 semitone)
    silent = vowel_frame(VOWELS["oo"], 196, ga=0, vp=6, vr=70)
    sung = vowel_frame(VOWELS["oo"], 196, ga=70, vp=6, vr=70)
    return vt_target([(silent, 2), (sung, 100), (sung, 1300)])


def make_vt_seven():
    from vocaltract.reference import singing_demo_seven

    demo = singing_demo_seven.build()
    tenor = [f for f in demo.frames if f.tract == 1][:18]          # first note: "seven"
    frames, pace = [], 100
    for f in tenor:
        time = round(0x01000000 / f.step)                          # recover go() time at pace 100
        frames.append((list(f.params), time))
    return vt_target(frames, pace)


# ---------------------------------------------------------------------------
# Synthetic (non-VocalTract) source-filter vowel
# ---------------------------------------------------------------------------

def make_synthetic_vowel():
    f0, formants, bandwidths = 120.0, VOWELS["ah"], (90, 110, 170, 250)
    sr = 48_000
    n = int(1.05 * sr)
    src = np.zeros(n)
    src[(np.arange(0, n, sr / f0)).astype(int)] = 1.0
    y = signal.lfilter([1.0], [1, -1.9, 0.9025], src)             # glottal tilt (-12 dB/oct)
    for f, bw in zip(formants, bandwidths):
        r = np.exp(-np.pi * bw / sr)
        y = signal.lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / sr), r * r], y)
    y = signal.lfilter([1, -0.97], [1.0], y)                       # lip radiation
    raw = np.zeros((N_FRAMES, 13))
    raw[:, GP] = hz_to_gp(f0)
    raw[:, GA] = 60                                                # nominal level only
    raw[:, F1:F1 + 4] = np.array(formants) / 19.53
    return finish(y[int(0.05 * sr):], sr), raw


# ---------------------------------------------------------------------------
# Real speech: 1 s excerpts + level A analysis params
# ---------------------------------------------------------------------------

def find_phrase(audio, sr, phrase: str) -> float:
    """Start time (s) of ``phrase`` in the recording, from Whisper word timestamps."""
    from vocaltract.analysis.intelligibility import ASR_RATE, load_model

    x = signal.resample_poly(audio, ASR_RATE, sr).astype(np.float32)
    result = load_model("small.en").transcribe(x, language="en", word_timestamps=True,
                                               temperature=0.0, fp16=False,
                                               condition_on_previous_text=False)
    words = [(w["word"].strip(" ,.?!\"'").lower(), w["start"])
             for seg in result["segments"] for w in seg["words"]]
    target = phrase.lower().split()
    for i in range(len(words) - len(target) + 1):
        if [w for w, _ in words[i:i + len(target)]] == target:
            return words[i][1]
    raise ValueError(f"phrase {phrase!r} not found in: {' '.join(w for w, _ in words)}")


def make_speech(clip: str, phrase: str):
    from vocaltract.analysis.resynth import HOP_S, resynthesize

    audio, sr = sf.read(ROOT / "assets" / "recordings" / f"{clip}.ogg")
    start = max(0.0, find_phrase(audio, sr, phrase) - 0.05)
    lo = int(start * sr)
    excerpt = audio[lo:lo + sr]

    # Analyse the whole clip (tracking is better with context), then sample the
    # level A frames (frame k targets time (k+1)*HOP_S) at the excerpt's centres.
    res = resynthesize(audio, sr, clip)
    frame_t = (np.arange(len(res.frames)) + 1) * HOP_S
    centres = start + FRAME_CENTRES_S
    raw = np.stack([np.interp(centres, frame_t, res.frames[:, i].astype(float))
                    for i in range(13)], 1)
    return finish(excerpt, sr), raw, start


# ---------------------------------------------------------------------------

def save(name: str, audio: np.ndarray, raw: np.ndarray, kind: str, meta: dict) -> dict:
    assert audio.shape == (N_SAMPLES,) and raw.shape == (N_FRAMES, 13)
    sf.write(OUT / f"{name}.wav", audio, RATE, subtype="PCM_16")
    params = np.clip(raw / 255.0, 0.0, 1.0).astype(np.float32)
    np.savez(OUT / f"{name}.npz", params=params, params_raw=raw.astype(np.float32),
             frame_times=FRAME_CENTRES_S.astype(np.float32), param_names=np.array(PARAM_NAMES),
             sample_rate=RATE, hop=HOP, kind=kind)
    return {"name": name, "wav": f"{name}.wav", "params": f"{name}.npz", "params_kind": kind, **meta}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []

    audio, raw = make_synthetic_vowel()
    manifest.append(save("synthetic_vowel_ah", audio, raw, "ground_truth", {
        "difficulty": 1, "source": "scipy source-filter model (not VocalTract)",
        "description": "steady /a/, f0 120 Hz, formants 730/1090/2440/3400 Hz with natural "
                       "bandwidths, glottal tilt and lip radiation. ga is nominal."}))

    for name, maker, difficulty, desc in [
        ("vt_vowel_ah", make_vt_vowel, 2, "steady /a/ at 120 Hz"),
        ("vt_glide_ah_ee", make_vt_glide, 3, "/a/ -> /i/ formant glide with 110 -> 150 Hz rise"),
        ("vt_sung_oo_vibrato", make_vt_sung, 3, "sung /u/ at 196 Hz, vibrato 5.3 Hz +/-0.75 st "
                                                "(vp/vr: the Phase 1 source ignores vibrato)"),
        ("vt_seven", make_vt_seven, 4, "\"seven\" from SingingDemoSeven (tenor, first note): "
                                       "frication (fa/ff) and nasal (na/nf), which the Phase 1 "
                                       "source cannot produce"),
    ]:
        audio, raw = maker()
        manifest.append(save(name, finish(audio, VT_RATE), raw, "ground_truth", {
            "difficulty": difficulty, "source": "vocaltract.reference engine (20 kHz -> 16 kHz)",
            "description": desc}))

    for name, clip, phrase, desc in [
        ("speech_male", "5703-47212-0000", "during the summer months", "male reader, f0 ~80 Hz"),
        ("speech_female", "198-209-0000", "the next morning", "female reader, f0 ~210 Hz"),
    ]:
        audio, raw, start = make_speech(clip, phrase)
        manifest.append(save(name, audio, raw, "analysis", {
            "difficulty": 5, "source": f"assets/recordings/{clip}.ogg from {start:.2f} s "
                                       "(LibriSpeech, CC BY 4.0)",
            "description": f"\"{phrase}...\", {desc}. Params are level A analysis "
                           "(vocaltract.analysis.resynth), an approximate initialization, "
                           "not ground truth."}))

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for m in manifest:
        print(f"{m['difficulty']}  {m['name']:20s} {m['params_kind']:12s} {m['description'][:70]}")


if __name__ == "__main__":
    main()
