"""
Differentiable polyphase resampling, matching ``scipy.signal.resample_poly``.

Used to render at VocalTract's native 20 kHz and deliver 16 kHz audio, exactly
the way the Phase 1 VocalTract targets were produced
(``scripts/make_phase1_assets.py``: engine render, then ``resample_poly``).
The filter taps are designed once with scipy (Kaiser window, beta 5, as scipy
does) and applied with ``conv1d``, so gradients flow to the input.
"""

from fractions import Fraction
from functools import lru_cache

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F


@lru_cache(maxsize=None)
def _taps(up: int, down: int) -> np.ndarray:
    """scipy.signal.resample_poly's default low-pass filter for (up, down)."""
    max_rate = max(up, down)
    half_len = 10 * max_rate
    return scipy.signal.firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * up


def resample_poly(x: torch.Tensor, up: int, down: int) -> torch.Tensor:
    """
    Resample the last dimension by ``up / down`` (zero-phase FIR, zero-padded edges).

    Args:
        x: Signal (..., T).
        up: Upsampling factor.
        down: Downsampling factor.

    Returns:
        (..., ceil(T * up / down)), same dtype as ``x``; equal to
        ``scipy.signal.resample_poly(x, up, down)`` up to rounding.
    """
    ratio = Fraction(up, down)
    up, down = ratio.numerator, ratio.denominator
    if up == down == 1:
        return x
    taps = torch.as_tensor(_taps(up, down), dtype=x.dtype, device=x.device)
    half_len = (len(taps) - 1) // 2
    shape, length = x.shape[:-1], x.shape[-1]
    flat = x.reshape(-1, 1, length)

    stuffed = flat.new_zeros(flat.shape[0], 1, length * up)
    stuffed[..., ::up] = flat
    filtered = F.conv1d(F.pad(stuffed, (half_len, half_len)), taps.flip(0).view(1, 1, -1))
    out_length = -(-length * up // down)  # ceil
    return filtered[..., ::down][..., :out_length].reshape(*shape, out_length)
