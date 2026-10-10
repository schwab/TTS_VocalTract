"""
Placeholder excitation source for the Phase 1 pipeline.

This module produces the signal that drives the LPC filter:

    params (B,T_frames,13) -> make_excitation -> excitation (B,T_frames*hop)
                                              -> LPCFilter -> audio

Phase 1 deliberately uses a very simple source: a sine carrier at the glottal
pitch (``gp``) scaled by the glottal amplitude (``ga``), plus white noise scaled
by the aspiration amplitude (``aa``). Vibrato, frication and nasal parameters
are ignored here. Phase 2 replaces the sine with a real glottal pulse model.

Every step is differentiable with respect to ``params``:

1. ``gp`` is converted to Hz at frame rate via ``denormalize``, then the
   source controls are upsampled to sample rate with linear
   interpolation (``F.interpolate(mode="linear")``).
2. Instantaneous frequency is integrated with ``torch.cumsum`` so the phase
   stays continuous when pitch changes between frames.
3. ``torch.sin`` turns phase into the carrier.

The noise itself carries no gradient; only its gain ``aa`` does.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .params import PARAM_DIM, PARAM_NAMES, denormalize

GA_INDEX = PARAM_NAMES.index("ga")
AA_INDEX = PARAM_NAMES.index("aa")


def upsample_frames(frames: torch.Tensor, hop: int) -> torch.Tensor:
    """
    Linearly interpolate frame-rate values to sample rate.

    Args:
        frames: Tensor of shape (B, T_frames, C).
        hop: Number of samples per frame.

    Returns:
        Tensor of shape (B, T_frames * hop, C).
    """
    num_frames = frames.shape[1]
    upsampled = F.interpolate(
        frames.transpose(1, 2), size=num_frames * hop, mode="linear", align_corners=False
    )
    return upsampled.transpose(1, 2)


def sine_from_frequency(freq_hz: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Phase-continuous sine for a per-sample frequency track.

    Phase is accumulated in cycles and wrapped to [0, 1) before ``sin`` to
    keep float32 precision on long signals. The wrap (``floor``) has zero
    gradient, so the derivative of the phase is unchanged.

    Args:
        freq_hz: Instantaneous frequency of shape (B, T) in Hz.
        sample_rate: Sample rate in Hz.

    Returns:
        Sine wave of shape (B, T), starting at phase 0.
    """
    cycles = torch.cumsum(freq_hz / sample_rate, dim=-1)
    cycles = cycles - cycles[..., :1]  # start at phase 0
    wrapped = cycles - torch.floor(cycles)
    return torch.sin(2 * torch.pi * wrapped)


def make_excitation(
    params: torch.Tensor,
    sample_rate: int = 16000,
    hop: int = 320,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate a Phase 1 excitation signal from parameter frames.

    excitation = ga * sin(2*pi * integral(gp)) + aa * white_noise

    where ``gp`` is in Hz (via :func:`vocaltract.params.denormalize`) and
    ``ga``/``aa`` are the normalized [0, 1] amplitudes.

    Args:
        params: Normalized parameter frames of shape (B, T_frames, 13).
        sample_rate: Output sample rate in Hz.
        hop: Samples per parameter frame.
        generator: Optional RNG for the noise component (for reproducibility).

    Returns:
        Excitation of shape (B, T_frames * hop).
    """
    if params.ndim != 3 or params.shape[-1] != PARAM_DIM:
        raise ValueError(f"params must be (B, T_frames, {PARAM_DIM}), got {tuple(params.shape)}")

    # Pitch is converted to Hz at frame rate, then all source controls are
    # upsampled together. Amplitudes stay normalized to [0, 1].
    pitch_hz = denormalize(params)["gp"]
    source_frames = torch.stack(
        [pitch_hz, params[..., GA_INDEX], params[..., AA_INDEX]], dim=-1
    )
    pitch_hz, ga, aa = upsample_frames(source_frames, hop).unbind(dim=-1)

    voiced = ga * sine_from_frequency(pitch_hz, sample_rate)

    noise = torch.randn(
        voiced.shape, generator=generator, dtype=voiced.dtype, device=voiced.device
    )
    return voiced + aa * noise

