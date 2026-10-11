"""
Placeholder excitation source for the Phase 1 pipeline.

This module produces the signal that drives the LPC filter:

    params (B,T_frames,13) -> make_excitation -> excitation (B,T_frames*hop)
                                              -> LPCFilter -> audio

Phase 1 deliberately uses a simple source: a band-limited harmonic series at
the glottal pitch (``gp``) scaled by the glottal amplitude (``ga``), plus white
noise scaled by the aspiration amplitude (``aa``). Vibrato, frication and nasal
parameters are ignored here. Phase 2 replaces the harmonic series with a real
glottal pulse model.

Why harmonics and not a single sine: the LPC filter is linear, so it can only
reshape energy the source already has. A pure sine has energy only at f0, so
no filter can produce the 2*f0, 3*f0, ... harmonics that carry a vowel's
formants. The series here is sum_k sin(k * phase) / k for every k*f0 below
Nyquist (a band-limited sawtooth, -6 dB/octave, the classic DDSP-style
harmonic oscillator). ``max_harmonics=1`` gives back the pure sine.

Every step is differentiable with respect to ``params``:

1. ``gp`` is converted to Hz at frame rate via ``denormalize``, then the
   source controls are upsampled to sample rate with linear
   interpolation (``F.interpolate(mode="linear")``).
2. Instantaneous frequency is integrated with ``torch.cumsum`` so the phase
   stays continuous when pitch changes between frames.
3. ``torch.sin`` of each harmonic's phase (k times the fundamental's) builds
   the carrier. Harmonics fade to zero approaching Nyquist (no aliasing, no jumps).

The noise itself carries no gradient; only its gain ``aa`` does.

Waveforms
---------
``waveform="sawtooth"`` (default) is the series above. ``waveform="vocaltract"``
uses the same harmonic machinery with the Fourier coefficients of VocalTract's
own glottal pulse. In the engine (``reference/engine.py``) the pulse is
``sine(antilog(phase) - 2**30)``; with the ROM tables, ``antilog`` of a 32-bit
phase ``phi * 2**32`` is ``2**(16 + 16 phi)`` and the sine takes 2**32 as a
full turn, so

    g(phi) = sin(2 pi (2**(16 (phi - 1)) - 1/4)) = -cos(2 pi 2**(16 (phi - 1)))

flat at -1 for most of the period, then one fast cosine pulse in the last
~20%. It is smooth across the period boundary, so its harmonics decay
quickly. The DC term is dropped (the formant filter removes it anyway).

``waveform="vocaltract_aliased"`` evaluates ``g(phi)`` directly at each
sample, DC included, exactly as the engine does. It is not band-limited: at
VocalTract's 20 kHz rate its harmonics above 10 kHz alias, and that aliasing
is part of the engine's sound (it fills 4-8 kHz of the 16 kHz targets). Use it
with ``render_rate=20000`` in :mod:`vocaltract.pipeline`.
"""

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
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


# Scale so the band-limited sawtooth peaks near 1 (sum sin(k x)/k peaks at pi/2).
SAWTOOTH_NORM = 2.0 / torch.pi

# Harmonics fade out linearly over this band just below Nyquist. A hard cutoff
# would switch a harmonic fully on or off as pitch moves it across Nyquist,
# making the output (and the loss) discontinuous in pitch.
NYQUIST_TAPER_HZ = 500.0


WAVEFORMS = ("sawtooth", "vocaltract", "vocaltract_aliased")

# Grid used to take the Fourier series of the VocalTract pulse.
_PULSE_GRID = 1 << 15


def vocaltract_pulse(phase: np.ndarray) -> np.ndarray:
    """VocalTract's glottal pulse over one period, ``phase`` in [0, 1) (see module docstring)."""
    return -np.cos(2 * np.pi * np.exp2(16.0 * (phase - 1.0)))


@lru_cache(maxsize=None)
def _vocaltract_coefficients(num_harmonics: int) -> Tuple[np.ndarray, np.ndarray]:
    """(cos, sin) Fourier coefficients of :func:`vocaltract_pulse` for k = 1..num_harmonics."""
    spectrum = np.fft.rfft(vocaltract_pulse(np.arange(_PULSE_GRID) / _PULSE_GRID)) / _PULSE_GRID
    c = spectrum[1 : num_harmonics + 1]
    # g = sum_k 2 Re(c_k e^{j 2 pi k phi}) = sum_k a_k cos + b_k sin
    return 2 * c.real, -2 * c.imag


def harmonic_coefficients(
    waveform: str, num_harmonics: int, dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fourier coefficients (cos_k, sin_k), k = 1..num_harmonics, of a source waveform.

    The waveform is ``sum_k cos_k cos(2 pi k phase) + sin_k sin(2 pi k phase)``.
    """
    if waveform == "sawtooth":
        k = torch.arange(1, num_harmonics + 1, dtype=dtype)
        return torch.zeros_like(k), SAWTOOTH_NORM / k
    if waveform == "vocaltract":
        cos_k, sin_k = _vocaltract_coefficients(num_harmonics)
        return torch.as_tensor(cos_k, dtype=dtype), torch.as_tensor(sin_k, dtype=dtype)
    raise ValueError(f"unknown waveform {waveform!r}, expected one of {WAVEFORMS}")


def harmonics_from_frequency(
    freq_hz: torch.Tensor,
    sample_rate: int,
    max_harmonics: Optional[int] = None,
    waveform: str = "sawtooth",
) -> torch.Tensor:
    """
    Band-limited periodic waveform for a pitch track: by default the sawtooth
    series ``sum_k sin(k * phase) / k``, or VocalTract's glottal pulse.

    Harmonic k is weighted by a linear taper that reaches 0 at Nyquist (over
    the top ``NYQUIST_TAPER_HZ``), so the output never aliases and stays
    continuous in pitch. Phase is continuous across pitch changes (see
    :func:`sine_from_frequency`).

    Args:
        freq_hz: Fundamental frequency of shape (B, T) in Hz.
        sample_rate: Sample rate in Hz.
        max_harmonics: Upper limit on the number of harmonics. ``None`` means
            as many as fit below Nyquist; 1 with the sawtooth gives a pure sine.
        waveform: ``"sawtooth"`` or ``"vocaltract"`` (see module docstring).

    Returns:
        Signal of shape (B, T), peaking near 1.
    """
    if waveform not in WAVEFORMS:
        raise ValueError(f"unknown waveform {waveform!r}, expected one of {WAVEFORMS}")
    nyquist = sample_rate / 2
    num_harmonics = int(nyquist // freq_hz.detach().min().clamp_min(1.0).item()) + 1
    if max_harmonics is not None:
        num_harmonics = min(num_harmonics, max_harmonics)
    num_harmonics = max(num_harmonics, 1)

    cycles = torch.cumsum(freq_hz / sample_rate, dim=-1)
    cycles = cycles - cycles[..., :1]  # start at phase 0
    wrapped = cycles - torch.floor(cycles)  # (B, T)
    if waveform == "vocaltract_aliased":
        return -torch.cos(2 * torch.pi * torch.exp2(16.0 * (wrapped - 1.0)))
    wrapped = wrapped.unsqueeze(-1)  # (B, T, 1)

    k = torch.arange(1, num_harmonics + 1, dtype=freq_hz.dtype, device=freq_hz.device)
    harmonic_hz = freq_hz.unsqueeze(-1) * k  # (B, T, K)
    taper = ((nyquist - harmonic_hz) / NYQUIST_TAPER_HZ).clamp(0.0, 1.0)
    angle = 2 * torch.pi * k * wrapped
    if waveform == "sawtooth" and max_harmonics == 1:
        return (torch.sin(angle) * taper).sum(dim=-1)  # pure sine, unit amplitude
    cos_k, sin_k = (c.to(freq_hz.device) for c in
                    harmonic_coefficients(waveform, num_harmonics, freq_hz.dtype))
    if waveform == "sawtooth":
        partials = sin_k * torch.sin(angle)
    else:
        partials = cos_k * torch.cos(angle) + sin_k * torch.sin(angle)
    return (partials * taper).sum(dim=-1)


def make_excitation(
    params: torch.Tensor,
    sample_rate: int = 16000,
    hop: int = 320,
    generator: Optional[torch.Generator] = None,
    max_harmonics: Optional[int] = None,
    waveform: str = "sawtooth",
) -> torch.Tensor:
    """
    Generate a Phase 1 excitation signal from parameter frames.

    excitation = ga * harmonics(integral(gp)) + aa * white_noise

    where ``gp`` is in Hz (via :func:`vocaltract.params.denormalize`) and
    ``ga``/``aa`` are the normalized [0, 1] amplitudes.

    Args:
        params: Normalized parameter frames of shape (B, T_frames, 13).
        sample_rate: Output sample rate in Hz.
        hop: Samples per parameter frame.
        generator: Optional RNG for the noise component (for reproducibility).
        max_harmonics: Limit on harmonics (``None``: all below Nyquist; 1: pure sine).
        waveform: Voiced waveform, ``"sawtooth"`` or ``"vocaltract"``.

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

    voiced = ga * harmonics_from_frequency(pitch_hz, sample_rate, max_harmonics, waveform)

    noise = torch.randn(
        voiced.shape, generator=generator, dtype=voiced.dtype, device=voiced.device
    )
    return voiced + aa * noise

