"""
The Phase 1 synthesis chain in one place.

    params (B, T_frames, 13)
      -> ParamToLPC                      -> coeffs (B, T_frames, order)
      -> min_response                    -> per-frame normalization
      -> make_excitation * min_response  -> excitation (B, T_frames * hop)
      -> LPCFilter(excitation, coeffs)   -> audio (B, T_frames * hop)

The excitation is multiplied by min|A(e^jw)| (1 / the filter's peak gain), frame by frame, so the
strongest resonance always has unit gain. Without this, an untrained (or
diverging) mapper can produce filter gains of 1e10 and more, and the output
level says nothing about the source amplitudes ``ga`` and ``aa``.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .filter import LPCFilter, frames_to_samples, min_response
from .mapper import ParamToLPC
from .source import make_excitation


@dataclass
class SynthesisTrace:
    """Intermediate tensors from one synthesis pass (for inspection and tests)."""

    frame_coeffs: torch.Tensor  # (B, T_frames, order)
    frame_norm: torch.Tensor  # (B, T_frames) min|A| = 1 / peak gain
    excitation: torch.Tensor  # (B, T) normalized excitation fed to the filter
    sample_coeffs: torch.Tensor  # (B, T, order)
    audio: torch.Tensor  # (B, T)


def synthesize_trace(
    params: torch.Tensor,
    mapper: ParamToLPC,
    hop: int = 320,
    sample_rate: int = 16000,
    generator: Optional[torch.Generator] = None,
    normalize_gain: bool = True,
) -> SynthesisTrace:
    """Run the full chain and return every intermediate tensor."""
    frame_coeffs = mapper(params)
    excitation = make_excitation(params, sample_rate=sample_rate, hop=hop, generator=generator)

    frame_norm = min_response(frame_coeffs)
    if normalize_gain:
        excitation = excitation * frames_to_samples(frame_norm.unsqueeze(-1), hop).squeeze(-1)

    sample_coeffs = frames_to_samples(frame_coeffs, hop)
    # The mapper's coefficients are float64; filter at that precision, then
    # return audio in the caller's dtype.
    audio = LPCFilter()(excitation, sample_coeffs).to(params.dtype)
    return SynthesisTrace(frame_coeffs, frame_norm, excitation, sample_coeffs, audio)


def synthesize(
    params: torch.Tensor,
    mapper: ParamToLPC,
    hop: int = 320,
    sample_rate: int = 16000,
    generator: Optional[torch.Generator] = None,
    normalize_gain: bool = True,
) -> torch.Tensor:
    """params (B, T_frames, 13) -> audio (B, T_frames * hop)."""
    return synthesize_trace(params, mapper, hop, sample_rate, generator, normalize_gain).audio
