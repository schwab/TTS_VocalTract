"""
Diagnostics for a Phase 1 Step 7 overfit run: where the trained filter put its poles, and how far the
result's harmonic levels are from the target's.

Runs ``vocaltract.phase1.overfit_demo`` (same options, passed through), captures the trained mapper, and
prints:

- log-spectral distance before/after (the demo's own metric)
- the trained filter's poles at the middle frame (frequency / bandwidth, Hz)
- band levels relative to each clip's peak, result vs target
- harmonic-level RMS error below and above 4 kHz (``vocaltract.analysis.compare.harmonic_levels``)

Usage::

    python scripts/step7_diagnostics.py --freeze aa=0                 # defaults: vt_vowel_ah, order 10
    python scripts/step7_diagnostics.py --order 12 --freeze aa=0
    python scripts/step7_diagnostics.py --target vt_glide_ah_ee --freeze aa=0

Used for docs/reviews/phase1_step7_review.md.
"""

import argparse
import sys

import numpy as np
import torch

import vocaltract.phase1.overfit_demo as od
from vocaltract.analysis.compare import harmonic_levels
from vocaltract.analysis.features import track
from vocaltract.mapper import ParamToLPC

SR = od.SAMPLE_RATE
BANDS = [(0, 1000), (1000, 4000), (4000, 6000), (6000, 8000)]


def poles(coeffs: np.ndarray) -> list[tuple[float, float]]:
    """(frequency, bandwidth) in Hz of the upper-half-plane roots of 1 - sum a_i z^-i."""
    roots = np.roots(np.r_[1.0, -coeffs])
    roots = roots[np.imag(roots) >= -1e-9]
    return sorted((float(np.angle(z) * SR / (2 * np.pi)), float(-np.log(abs(z)) * SR / np.pi)) for z in roots)


def band_level(y: np.ndarray, lo: float, hi: float, skip: int = 1600) -> float:
    """RMS spectrum level in [lo, hi) re the spectral peak, dB (skips the onset)."""
    seg = y[skip:]
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    return float(20 * np.log10(np.sqrt(np.mean(S[(f >= lo) & (f < hi)] ** 2)) / S.max()))


def harmonic_error(result: np.ndarray, target: np.ndarray, split: float = 4000.0) -> tuple[float, float]:
    """RMS dB difference of mean harmonic levels (k*f0 of the target's pitch), below/above ``split``."""
    tr = track(target, SR, 0.01)
    f0 = np.nanmedian(tr.f0)
    mean = {}
    for name, y in (("result", result), ("target", target)):
        rows = [h for h in harmonic_levels(y, SR, tr.times, tr.f0, SR / 2) if h]
        n = min(len(h) for h in rows)
        mean[name] = 10 * np.log10(np.mean([10 ** (np.array(h[:n]) / 10) for h in rows], axis=0))
        mean[name] -= mean[name].max()
    n = min(len(mean["result"]), len(mean["target"]))
    d = mean["result"][:n] - mean["target"][:n]
    freqs = f0 * np.arange(1, n + 1)
    rms = lambda m: float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else float("nan")
    return rms(freqs < split), rms(freqs >= split)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], add_help=False)
    ap.add_argument("--order", type=int, default=10, help="LPC order of the mapper (default 10)")
    ap.add_argument("-h", "--help", action="store_true")
    args, demo_argv = ap.parse_known_args(argv)
    if args.help:
        print(__doc__)
        return 0

    captured = {}

    class CapturingMapper(ParamToLPC):
        def __init__(self, *a, **k):
            k["order"] = args.order
            super().__init__(*a, **k)
            captured["mapper"] = self

    od.ParamToLPC = CapturingMapper
    cfg = od.parse_args([*demo_argv, "--out", "/dev/null"])
    r = od.run(cfg)

    with torch.no_grad():
        coeffs = captured["mapper"](r.params)[0, r.params.shape[1] // 2].double().numpy()
    result, target = r.final_audio[0].numpy(), r.target[0].numpy()
    below, above = harmonic_error(result, target)

    print(f"{cfg.target}, order {args.order}, best step {r.best_step}, {r.seconds:.0f} s")
    print(f"  LSD {od.log_spectral_distance(r.initial_audio, r.target):.1f} -> "
          f"{od.log_spectral_distance(r.final_audio, r.target):.1f} dB")
    print("  poles at the middle frame (Hz / bandwidth Hz): "
          + ", ".join(f"{f:.0f}/{b:.0f}" for f, b in poles(coeffs)))
    print("  band level re peak, result / target: "
          + "  ".join(f"{lo / 1000:g}-{hi / 1000:g} kHz {band_level(result, lo, hi):.0f}/{band_level(target, lo, hi):.0f}"
                      for lo, hi in BANDS))
    print(f"  harmonic-level RMS error: {below:.1f} dB below 4 kHz, {above:.1f} dB above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
