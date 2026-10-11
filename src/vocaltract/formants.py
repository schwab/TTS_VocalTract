"""
Measure formant frequencies from audio, for evaluating synthesis.

The spectral losses and log-spectral distance score how close two spectra are
overall; they do not check that formants are where they should be. A model can
lower both by holding one compromise vowel while the target glides between
vowels (this happened on ``vt_glide_ah_ee`` with the MLP mapper). This module
estimates F1..Fn frame by frame so tests and demos can check formant motion
directly.

Method: classic autocorrelation LPC per frame (pre-emphasis 0.97, Hann window,
order 14 at 16 kHz), then the angles of the complex roots of the LPC
polynomial. Roots below MIN_FORMANT_HZ or wider than MAX_BANDWIDTH_HZ are
discarded as non-formant poles (spectral tilt, noise). Not differentiable;
numpy only.
"""

from typing import Union

import numpy as np
import scipy.linalg
import torch

PRE_EMPHASIS = 0.97
MIN_FORMANT_HZ = 150.0
MAX_BANDWIDTH_HZ = 500.0


def measure_formants(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_formants: int = 3,
    order: int = 14,
    frame_length: int = 400,
    hop: int = 320,
) -> np.ndarray:
    """
    Estimate formant frequencies frame by frame.

    Args:
        audio: Mono signal, shape (T,) or (1, T).
        sample_rate: Sample rate in Hz.
        n_formants: Number of formants to return (lowest first).
        order: LPC analysis order.
        frame_length: Analysis window in samples (25 ms at 16 kHz).
        hop: Frame step in samples (320 matches the parameter frame rate).

    Returns:
        Array (frames, n_formants) in Hz. Frame i starts at sample i * hop.
        NaN where a frame is silent or has fewer than ``n_formants`` formant
        candidates.
    """
    x = torch.as_tensor(audio).detach().double().reshape(-1).numpy()
    x = np.append(x[0], x[1:] - PRE_EMPHASIS * x[:-1])
    window = np.hanning(frame_length)
    starts = range(0, len(x) - frame_length + 1, hop)
    result = np.full((len(starts), n_formants), np.nan)
    for i, start in enumerate(starts):
        frame = x[start : start + frame_length] * window
        r = np.correlate(frame, frame, "full")[frame_length - 1 : frame_length + order]
        if r[0] <= 1e-10:
            continue
        r[0] *= 1 + 1e-9  # tiny white-noise floor keeps the system positive definite
        a = scipy.linalg.solve_toeplitz(r[:order], r[1 : order + 1])
        roots = np.roots(np.concatenate([[1.0], -a]))
        roots = roots[roots.imag > 0]
        freqs = np.angle(roots) * sample_rate / (2 * np.pi)
        bandwidths = -np.log(np.abs(roots)) * sample_rate / np.pi
        keep = np.sort(freqs[(freqs > MIN_FORMANT_HZ) & (bandwidths < MAX_BANDWIDTH_HZ)])
        if len(keep) >= n_formants:
            result[i] = keep[:n_formants]
    return result


def formant_error_hz(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_formants: int = 3,
) -> np.ndarray:
    """
    Mean absolute per-frame formant error between two signals, per formant.

    Frames where either signal has no estimate are skipped.

    Returns:
        Array (n_formants,) in Hz (NaN if no frame has estimates in both).
    """
    p = measure_formants(pred, sample_rate, n_formants)
    t = measure_formants(target, sample_rate, n_formants)
    both = ~(np.isnan(p).any(-1) | np.isnan(t).any(-1))
    if not both.any():
        return np.full(n_formants, np.nan)
    return np.abs(p[both] - t[both]).mean(axis=0)
