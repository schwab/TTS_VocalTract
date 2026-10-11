"""
The Phase 1 synthesis chain in one place.

    params (B, T_frames, 13)
      -> mapper (FormantToLPC/ParamToLPC) -> coeffs (B, T_frames, order)
      -> mapper.numerator (if any)        -> FIR taps (B, T_frames, K)
      -> min_response                    -> per-frame normalization
      -> make_excitation * min_response  -> excitation (B, T_frames * hop)
      -> fir_frames(excitation, taps)    -> (only with a numerator)
      -> LPCFilter(excitation, coeffs)   -> audio (B, T_frames * hop)

With ``render_rate`` (e.g. VocalTract's 20000 Hz), the whole chain runs at that
rate with ``hop * render_rate / sample_rate`` samples per frame, and the result
is resampled to ``sample_rate`` with :func:`vocaltract.resample.resample_poly`.
The mapper must be built for the render rate.

The excitation is multiplied by min|A(e^jw)| (1 / the filter's peak gain), frame by frame, so the
strongest resonance always has unit gain. Without this, an untrained (or
diverging) mapper can produce filter gains of 1e10 and more, and the output
level says nothing about the source amplitudes ``ga`` and ``aa``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .filter import LPCFilter, fir_frames, frames_to_samples, min_response
from .resample import resample_poly
from .source import make_excitation


@dataclass
class SynthesisTrace:
    """Intermediate tensors from one synthesis pass (for inspection and tests)."""

    frame_coeffs: torch.Tensor  # (B, T_frames, order)
    frame_norm: torch.Tensor  # (B, T_frames) min|A| = 1 / peak gain
    excitation: torch.Tensor  # (B, T) normalized excitation fed to the filter
    sample_coeffs: torch.Tensor  # (B, T, order)
    audio: torch.Tensor  # (B, T) at sample_rate
    frame_numerator: Optional[torch.Tensor] = None  # (B, T_frames, K) FIR taps, if the mapper has them


def synthesize_trace(
    params: torch.Tensor,
    mapper: nn.Module,
    hop: int = 320,
    sample_rate: int = 16000,
    generator: Optional[torch.Generator] = None,
    normalize_gain: bool = True,
    waveform: str = "sawtooth",
    render_rate: Optional[int] = None,
) -> SynthesisTrace:
    """
    Run the full chain and return every intermediate tensor.

    ``waveform``: see :func:`make_excitation`. ``render_rate``: run the chain at
    this rate and resample to ``sample_rate`` (None: render at ``sample_rate``).
    Tensors other than ``audio`` are at the render rate.
    """
    render_rate = render_rate or sample_rate
    if (hop * render_rate) % sample_rate:
        raise ValueError(f"hop {hop} at {sample_rate} Hz is not a whole number of samples "
                         f"at {render_rate} Hz")
    mapper_rate = getattr(mapper, "sample_rate", render_rate)
    if mapper_rate != render_rate:
        raise ValueError(f"mapper is built for {mapper_rate} Hz but the chain renders at {render_rate} Hz")
    out_hop, hop = hop, hop * render_rate // sample_rate

    frame_coeffs = mapper(params)
    numerator_fn = getattr(mapper, "numerator", None)
    frame_numerator = numerator_fn(params) if numerator_fn is not None else None
    excitation = make_excitation(
        params, sample_rate=render_rate, hop=hop, generator=generator, waveform=waveform
    )

    frame_norm = min_response(frame_coeffs, numerator=frame_numerator)
    if normalize_gain:
        excitation = excitation * frames_to_samples(frame_norm.unsqueeze(-1), hop).squeeze(-1)
    if frame_numerator is not None:
        excitation = fir_frames(excitation, frames_to_samples(frame_numerator, hop))

    sample_coeffs = frames_to_samples(frame_coeffs, hop)
    # The mapper's coefficients are float64; filter at that precision, then
    # return audio in the caller's dtype.
    audio = LPCFilter()(excitation, sample_coeffs)
    if render_rate != sample_rate:
        audio = resample_poly(audio, sample_rate, render_rate)[..., : params.shape[1] * out_hop]
    audio = audio.to(params.dtype)
    return SynthesisTrace(frame_coeffs, frame_norm, excitation, sample_coeffs, audio, frame_numerator)


def synthesize(
    params: torch.Tensor,
    mapper: nn.Module,
    hop: int = 320,
    sample_rate: int = 16000,
    generator: Optional[torch.Generator] = None,
    normalize_gain: bool = True,
    waveform: str = "sawtooth",
    render_rate: Optional[int] = None,
) -> torch.Tensor:
    """params (B, T_frames, 13) -> audio (B, T_frames * hop) at ``sample_rate``."""
    return synthesize_trace(
        params, mapper, hop, sample_rate, generator, normalize_gain, waveform, render_rate
    ).audio
