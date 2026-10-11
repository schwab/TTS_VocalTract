"""
Differentiable time-varying LPC (all-pole) synthesis filter.

This module is the "vocal tract" stage of the Phase 1 pipeline:

    params (B,T_frames,13) -> ParamToLPC -> coeffs
    excitation (B,T)       -> LPCFilter(excitation, coeffs) -> audio (B,T)

It is a thin wrapper around ``torchlpc.sample_wise_lpc``. ``torchlpc`` runs the
recursive filter with efficient custom kernels (Numba on CPU, CUDA on GPU) and
provides gradients by running a *reverse filtering operation* (the backward
pass of an all-pole filter is itself an all-pole filter run backwards in
time), instead of letting autograd unroll the recursion sample by sample.

Sign convention
---------------
Coefficients follow the same convention as :class:`vocaltract.mapper.ParamToLPC`:
the synthesis filter is

    H(z) = 1 / (1 - sum_{i=1}^{order} a_i z^{-i})
    y[n] = x[n] + sum_{i=1}^{order} a_i[n] * y[n - i]

``torchlpc`` uses the opposite sign (``1 + sum A_i z^{-i}``), so the wrapper
negates the coefficients before calling it. This keeps the mapper's
pole-stability guarantee valid for the filter.

Fallback
--------
If ``torchlpc`` cannot be imported, a naive pure-PyTorch IIR loop is used
instead (see :func:`naive_sample_wise_lpc`). It is numerically equivalent but
runs one Python iteration per sample, which is orders of magnitude slower and
builds a very large autograd graph — fine for tests, not for training.
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    import torchlpc

    HAS_TORCHLPC = True
except ImportError:  # pragma: no cover - exercised only on platforms without torchlpc
    torchlpc = None
    HAS_TORCHLPC = False


def naive_sample_wise_lpc(excitation: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Reference all-pole filter implemented as an explicit per-sample loop.

    Uses the ``1 - sum a_i z^{-i}`` convention (same as :class:`LPCFilter`)
    with zero initial state. Differentiable through ordinary autograd.

    Args:
        excitation: Source signal of shape (B, T).
        coeffs: Time-varying LPC coefficients of shape (B, T, order).

    Returns:
        Filtered signal of shape (B, T).
    """
    batch, num_samples = excitation.shape
    order = coeffs.shape[-1]

    # history[:, i] holds y[n - 1 - i]; start from zero initial state
    history = excitation.new_zeros(batch, order)
    outputs = []
    for n in range(num_samples):
        y_n = excitation[:, n] + (coeffs[:, n] * history).sum(dim=-1)
        outputs.append(y_n)
        history = torch.cat([y_n.unsqueeze(-1), history[:, :-1]], dim=-1)
    return torch.stack(outputs, dim=-1)


def frames_to_samples(coeffs: torch.Tensor, hop: int) -> torch.Tensor:
    """
    Hold frame-rate LPC coefficients constant across each frame's samples.

    Direct-form LPC coefficients are *not* interpolated: a linear blend of two
    stable coefficient sets can have poles outside the unit circle. Holding
    each frame keeps every sample on a set the mapper guarantees is stable.

    Args:
        coeffs: Frame-rate coefficients of shape (B, T_frames, order).
        hop: Samples per frame.

    Returns:
        Sample-rate coefficients of shape (B, T_frames * hop, order).
    """
    return coeffs.repeat_interleave(hop, dim=1)


def fir_frames(signal: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """
    Time-varying FIR filter ``y[n] = sum_k taps[n, k] * signal[n - k]`` (zero initial state).

    Args:
        signal: (B, T).
        taps: Per-sample taps (B, T, K), e.g. frame taps held by :func:`frames_to_samples`.

    Returns:
        (B, T) at the promoted dtype of the inputs.
    """
    dtype = torch.promote_types(signal.dtype, taps.dtype)
    num_taps = taps.shape[-1]
    padded = torch.nn.functional.pad(signal.to(dtype), (num_taps - 1, 0))
    history = padded.unfold(-1, num_taps, 1).flip(-1)  # (B, T, K): signal[n - k]
    return (history * taps.to(dtype)).sum(-1)


def min_response(
    coeffs: torch.Tensor, n_fft: int = 512, numerator: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Minimum of ``|A(e^{jw})|`` over an ``n_fft``-point frequency grid.

    This is the reciprocal of the all-pole filter's peak gain. Multiplying the
    excitation by it normalizes the filter so its strongest resonance has unit
    gain. Multiplying (rather than dividing by :func:`peak_gain`) keeps the
    gradient bounded: d(1/x)/dx = -1/x**2 overflows float32 for sharp
    resonances where x ~ 1e-7.

    With a ``numerator`` N (pole-zero filter N/A), returns ``min |A| / |N|``,
    again the reciprocal of the peak gain. The minimum sits at a resonance
    peak, where |N| is far from 0, so the clamp below never shapes the result.

    Args:
        coeffs: LPC coefficients of shape (..., order), ``1 - sum`` convention.
        n_fft: Frequency-grid size; must exceed ``order``.
        numerator: Optional FIR numerator coefficients (..., K).

    Returns:
        Tensor of shape (...).
    """
    polynomial = torch.cat([torch.ones_like(coeffs[..., :1]), -coeffs], dim=-1)
    response = torch.fft.rfft(polynomial, n=n_fft).abs()
    if numerator is not None:
        response = response / torch.fft.rfft(numerator, n=n_fft).abs().clamp_min(1e-12)
    return response.min(dim=-1).values


def peak_gain(
    coeffs: torch.Tensor, n_fft: int = 512, numerator: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Peak magnitude response of the all-pole (or pole-zero) filter (for reporting)."""
    return 1.0 / min_response(coeffs, n_fft, numerator).clamp_min(1e-12)


class LPCFilter(nn.Module):
    """
    Time-varying all-pole LPC synthesis filter.

    Wraps ``torchlpc.sample_wise_lpc``, which uses efficient custom kernels
    and computes gradients through a reverse filtering operation, so gradients
    flow to both the excitation and the coefficients.

    Phase 1 always starts from a zero initial state.

    Args:
        use_torchlpc: Use ``torchlpc`` when True (default) and available. When
            False, or when ``torchlpc`` is not installed, falls back to the
            slow :func:`naive_sample_wise_lpc` loop.
    """

    def __init__(self, use_torchlpc: bool = True):
        super().__init__()
        self.use_torchlpc = use_torchlpc and HAS_TORCHLPC

    def forward(self, excitation: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Filter an excitation signal with per-sample LPC coefficients.

        Args:
            excitation: Source signal of shape (B, T).
            coeffs: LPC coefficients of shape (B, T, order) in the
                ``1 - sum a_i z^{-i}`` convention (as produced by ParamToLPC).

        Filtering runs at the higher precision of the two inputs, so float64
        coefficients (as ParamToLPC produces) are never rounded to float32:
        sharp resonances are not stable after that rounding.

        Returns:
            Synthesized audio of shape (B, T), same dtype as ``excitation``.
        """
        self._check_shapes(excitation, coeffs)
        compute_dtype = torch.promote_types(excitation.dtype, coeffs.dtype)
        x, a = excitation.to(compute_dtype), coeffs.to(compute_dtype)

        if self.use_torchlpc:
            # torchlpc expects the 1 + sum A_i z^{-i} convention: negate.
            y = torchlpc.sample_wise_lpc(x, -a)
        else:
            y = naive_sample_wise_lpc(x, a)
        return y.to(excitation.dtype)

    @staticmethod
    def _check_shapes(excitation: torch.Tensor, coeffs: torch.Tensor) -> None:
        if excitation.ndim != 2:
            raise ValueError(f"excitation must be (B, T), got shape {tuple(excitation.shape)}")
        if coeffs.ndim != 3:
            raise ValueError(f"coeffs must be (B, T, order), got shape {tuple(coeffs.shape)}")
        if coeffs.shape[:2] != excitation.shape:
            raise ValueError(
                f"coeffs leading dims {tuple(coeffs.shape[:2])} must match "
                f"excitation shape {tuple(excitation.shape)}"
            )

    @staticmethod
    @torch.no_grad()
    def analyze(audio: torch.Tensor, order: int, frame_length: Optional[int] = None) -> torch.Tensor:
        """
        Estimate LPC coefficients from audio, for reference comparison only.

        Not differentiable (runs under ``torch.no_grad``). Coefficients are
        returned in the same ``1 - sum a_i z^{-i}`` convention as ``forward``.

        Args:
            audio: Signal of shape (B, T).
            order: LPC order.
            frame_length: Analysis frame length in samples. Defaults to the
                whole signal (one frame).

        Returns:
            LPC coefficients of shape (B, num_frames, order).
        """
        frame_length = frame_length or audio.shape[-1]
        frames = audio.unfold(-1, frame_length, frame_length)  # (B, F, L)
        autocorr = _autocorrelation(frames, order)
        return -_levinson_from_autocorr(autocorr)


def _autocorrelation(frames: torch.Tensor, order: int) -> torch.Tensor:
    """Biased autocorrelation r[0..order] along the last dim: (..., L) -> (..., order+1)."""
    length = frames.shape[-1]
    return torch.stack(
        [(frames[..., : length - lag] * frames[..., lag:]).sum(-1) for lag in range(order + 1)],
        dim=-1,
    )


def _levinson_from_autocorr(r: torch.Tensor) -> torch.Tensor:
    """
    Classic Levinson-Durbin solve of the normal equations.

    Returns coefficients in the ``1 + sum A_i z^{-i}`` convention, shape (..., order).
    """
    order = r.shape[-1] - 1
    a = r.new_zeros(*r.shape[:-1], 0)
    error = r[..., 0].clamp_min(1e-12)
    for m in range(1, order + 1):
        acc = r[..., m] + (a * r[..., 1:m].flip(-1)).sum(-1)
        k = -acc / error
        a = torch.cat([a + k.unsqueeze(-1) * a.flip(-1), k.unsqueeze(-1)], dim=-1)
        error = (error * (1 - k.pow(2))).clamp_min(1e-12)
    return a
