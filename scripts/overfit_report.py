"""
Score a folder of overfit-demo outputs: harmonic-level error before/after, and optionally Whisper transcripts.

For each ``<prefix><name>_target.wav`` with matching ``<prefix><name>_initial.wav`` and ``<prefix><name>.wav``,
prints the harmonic-level RMS error (dB) of the initial and final audio against the target, below and above
4 kHz. Levels are sampled at k*f0 along the *target's* Praat pitch track, interpolated onto a 20 Hz grid,
power-averaged over voiced frames after 50 ms, and each envelope is normalized to its own peak. This is the same
measure as the compare tool's harmonic-levels pane.

Usage::

    python scripts/overfit_report.py ~/project/TTS_VocalTract/out/fm --prefix final_ --asr speech

Used for docs/reviews/phase1_completion_review.md.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from vocaltract.analysis.compare import harmonic_levels
from vocaltract.analysis.features import track

STEP_HZ = 20.0
SPLIT_HZ = 4000.0


def envelope(y: np.ndarray, sr: int, times: np.ndarray, f0: np.ndarray, skip_s: float = 0.05) -> np.ndarray:
    """Harmonic envelope on a STEP_HZ grid (dB re its peak; NaN where no harmonic data)."""
    n = int(sr / 2 / STEP_HZ) + 1
    acc, cnt = np.zeros(n), np.zeros(n)
    for t, f, h in zip(times, f0, harmonic_levels(y, sr, times, f0, sr / 2)):
        if not h or t < skip_s:
            continue
        h = np.asarray(h, dtype=float)
        g = np.arange(int(np.ceil(f / STEP_HZ)), n)
        pos = g * STEP_HZ / f - 1
        ok = pos < len(h) - 1
        g, pos = g[ok], pos[ok]
        k = pos.astype(int)
        acc[g] += 10 ** ((h[k] + (h[k + 1] - h[k]) * (pos - k)) / 10)
        cnt[g] += 1
    env = np.full(n, np.nan)
    env[cnt > 0] = 10 * np.log10(acc[cnt > 0] / cnt[cnt > 0])
    return env - np.nanmax(env)


def rms_error(y: np.ndarray, target_env: np.ndarray, sr: int, tr) -> tuple[float, float]:
    d = envelope(y, sr, tr.times, tr.f0) - target_env
    f = np.arange(len(d)) * STEP_HZ
    rms = lambda m: float(np.sqrt(np.nanmean(d[m] ** 2)))
    return rms(f < SPLIT_HZ), rms(f >= SPLIT_HZ)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--prefix", default="", help="file prefix, e.g. final_")
    ap.add_argument("--asr", nargs="?", const="", metavar="SUBSTRING",
                    help="also transcribe with Whisper (only names containing SUBSTRING, if given)")
    args = ap.parse_args(argv)

    targets = sorted(args.folder.glob(f"{args.prefix}*_target.wav"))
    print(f"{'clip':22s} harmonic RMS error, dB (< 4 kHz / > 4 kHz)")
    for tpath in targets:
        name = tpath.name[len(args.prefix):-len("_target.wav")]
        initial = args.folder / f"{args.prefix}{name}_initial.wav"
        final = args.folder / f"{args.prefix}{name}.wav"
        if not (initial.exists() and final.exists()):
            continue
        target, sr = sf.read(tpath)
        tr = track(target, sr, 0.01)
        tenv = envelope(target, sr, tr.times, tr.f0)
        (ib, ia), (fb, fa) = (rms_error(sf.read(p)[0], tenv, sr, tr) for p in (initial, final))
        print(f"{name:22s} initial {ib:4.1f} / {ia:4.1f}   final {fb:4.1f} / {fa:4.1f}")
        if args.asr is not None and args.asr in name:
            from vocaltract.analysis.intelligibility import transcribe
            for label, p in (("target", tpath), ("initial", initial), ("final", final)):
                y, s = sf.read(p)
                print(f"    whisper {label:7s}: {transcribe(y, s)!r}")


if __name__ == "__main__":
    main()
