"""
Acoustic feature tracking for analysis/resynthesis (README "level A").

Measures, on a fixed frame grid, the quantities that map onto VocalTract's
13 parameters:

- **f0** (Praat autocorrelation pitch) -> ``gp``; NaN where unvoiced
- **F1..F4** (Praat Burg formants) -> ``f1``..``f4``
- **low-band RMS** (< ``split_hz``) -> level of the voiced/aspirated part
  that VocalTract sends through its formant resonators (``ga``/``aa``)
- **high-band RMS and centroid** (> ``split_hz``) -> frication (``fa``/``ff``),
  which VocalTract adds *after* the resonators as noise-jittered sine

Praat is used through ``parselmouth`` (``pip install praat-parselmouth``).
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class Track:
    """Per-frame features. All arrays have length ``len(times)``."""

    times: np.ndarray      # seconds, analysis instants
    f0: np.ndarray         # Hz, NaN when unvoiced
    formants: np.ndarray   # (T, 4) Hz, NaN where Praat found none
    low_rms: np.ndarray    # RMS below split_hz
    high_rms: np.ndarray   # RMS above split_hz
    high_centroid: np.ndarray  # Hz, spectral centroid above split_hz
    max_formant: float     # Praat ceiling used for formant search

    @property
    def voiced(self) -> np.ndarray:
        return ~np.isnan(self.f0)


def _frame_rms(x: np.ndarray, sr: int, times: np.ndarray, width: float) -> np.ndarray:
    """RMS in a ``width``-second window centred on each time."""
    csum = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    half = max(1, int(width * sr / 2))
    centre = np.round(times * sr).astype(int)
    lo = np.clip(centre - half, 0, len(x))
    hi = np.clip(centre + half, 0, len(x))
    n = np.maximum(hi - lo, 1)
    return np.sqrt((csum[hi] - csum[lo]) / n)


def _high_centroid(x: np.ndarray, sr: int, times: np.ndarray, split_hz: float,
                   width: float) -> np.ndarray:
    nfft = 1 << int(np.ceil(np.log2(width * sr)))
    padded = np.pad(x, nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = freqs >= split_hz
    win = np.hanning(nfft)
    out = np.empty(len(times))
    for i, t in enumerate(times):
        c = int(t * sr) + nfft
        seg = padded[c - nfft // 2:c + nfft // 2] * win
        p = np.abs(np.fft.rfft(seg))[band] ** 2
        out[i] = (freqs[band] * p).sum() / p.sum() if p.sum() > 0 else np.nan
    return out


def track(audio: np.ndarray, sr: int, hop: float, *, f0_min: float = 60.0,
          f0_max: float = 500.0, max_formant: float | None = None,
          split_hz: float = 2500.0) -> Track:
    """Track features every ``hop`` seconds over a mono recording.

    ``max_formant`` defaults by voice: 5000 Hz if the median f0 is below
    160 Hz (typical adult male), else 5500 Hz (Praat's usual advice).
    """
    import parselmouth

    audio = np.asarray(audio, dtype=np.float64)
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    times = np.arange(int(len(audio) / sr / hop)) * hop

    pitch = snd.to_pitch_ac(time_step=hop, pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0 = np.array([pitch.get_value_at_time(t) for t in times], dtype=np.float64)
    f0[f0 <= 0] = np.nan

    if max_formant is None:
        voiced_f0 = f0[~np.isnan(f0)]
        max_formant = 5000.0 if voiced_f0.size and np.median(voiced_f0) < 160 else 5500.0
    formant = snd.to_formant_burg(time_step=hop, max_number_of_formants=5,
                                  maximum_formant=max_formant)
    formants = np.array([[formant.get_value_at_time(n, t) for n in range(1, 5)] for t in times],
                        dtype=np.float64)

    nyq = sr / 2
    sos_lo = signal.butter(6, min(split_hz, 0.95 * nyq) / nyq, "low", output="sos")
    sos_hi = signal.butter(6, min(split_hz, 0.95 * nyq) / nyq, "high", output="sos")
    width = 2 * hop
    low_rms = _frame_rms(signal.sosfiltfilt(sos_lo, audio), sr, times, width)
    high_rms = _frame_rms(signal.sosfiltfilt(sos_hi, audio), sr, times, width)
    centroid = _high_centroid(audio, sr, times, split_hz, max(width, 0.016))

    return Track(times=times, f0=f0, formants=formants, low_rms=low_rms, high_rms=high_rms,
                 high_centroid=centroid, max_formant=max_formant)
