"""
Harmonic-level error: how far a signal's harmonic envelope is from a target's.

Levels are sampled at k*f0 along the *target's* Praat pitch track
(:func:`vocaltract.analysis.compare.harmonic_levels`), interpolated onto a
20 Hz grid, power-averaged over voiced frames after the first 50 ms, and each
envelope is normalized to its own peak. The error is the RMS difference of the
two envelopes below and above 4 kHz. This is the same measure as the compare
tool's harmonic-levels pane. Over a whole utterance it compares long-term
average envelopes, so it shows spectral balance, not per-phoneme accuracy.
"""

import numpy as np

from .compare import harmonic_levels
from .features import Track, track

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


def rms_error(y: np.ndarray, target_env: np.ndarray, sr: int, tr: Track) -> tuple[float, float]:
    """RMS envelope difference (dB) below and above SPLIT_HZ, along the target's track ``tr``."""
    d = envelope(y, sr, tr.times, tr.f0) - target_env
    f = np.arange(len(d)) * STEP_HZ
    rms = lambda m: float(np.sqrt(np.nanmean(d[m] ** 2)))
    return rms(f < SPLIT_HZ), rms(f >= SPLIT_HZ)


class HarmonicReference:
    """A target's pitch track and harmonic envelope, to score several signals against."""

    def __init__(self, target: np.ndarray, sr: int, hop_s: float = 0.01):
        self.sr = sr
        self.track = track(target, sr, hop_s)
        self.envelope = envelope(target, sr, self.track.times, self.track.f0)

    def error(self, y: np.ndarray) -> tuple[float, float]:
        """(dB below 4 kHz, dB above 4 kHz)."""
        return rms_error(y, self.envelope, self.sr, self.track)
