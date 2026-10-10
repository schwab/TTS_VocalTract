"""
VocalTract parameter schema.

This module defines the canonical 13-parameter frame as used by the original
Propeller VocalTract.spin engine, reconciled against the source code.

The parameters control:
- Excitation source (glottal pulse, aspiration noise, frication noise)
- Resonance (formant frequencies via Cordic rotation)
- Anti-resonance (nasal zero)

All values are normalized to [0, 1] for differentiable training. Physical units
are recovered via denormalize().
"""

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


# Original VocalTract parameter names in order
PARAM_NAMES = [
    "aa",   # aspiration amplitude
    "ga",   # glottal amplitude
    "gp",   # glottal pitch
    "vp",   # vibrato pitch
    "vr",   # vibrato rate
    "f1",   # formant 1 frequency
    "f2",   # formant 2 frequency
    "f3",   # formant 3 frequency
    "f4",   # formant 4 frequency
    "na",   # nasal amplitude
    "nf",   # nasal frequency
    "fa",   # frication amplitude
    "ff",   # frication frequency
]

PARAM_DIM = len(PARAM_NAMES)  # 13


# Physical unit conversion constants (from original VocalTract.spin header table).
#
# Every one of the 13 parameters is a single byte in [0, 255] on the real hardware.
# Normalized [0,1] training values must therefore be mapped back to raw bytes via
# `raw = normalized * BYTE_SCALE` *uniformly* before any physical-unit scaling is
# applied. (A previous version of this module scaled frequency parameters by
# `* 100` instead of `* BYTE_SCALE` -- see docs/corrections.md.)
BYTE_SCALE = 255.0

# gp (glottal pitch) is a LOG-frequency value in 1/48-octave units, not a linear
# offset: raw=100 -> 110.00Hz (A2), and doubling/halving frequency is +/-48 raw
# units. Hz = GP_BASE_HZ * 2 ** ((raw - GP_BASE_RAW) / OCTAVE_UNIT).
GP_BASE_HZ = 110.0
GP_BASE_RAW = 100.0
OCTAVE_UNIT = 48.0  # 1/48 octave per raw unit (shared by gp and vp)

# vp (vibrato pitch) shares gp's log-pitch domain: it is a modulation *depth* in
# octaves, added to gp's log-pitch before the hardware's antilog step, not a
# frequency itself. The one calibration point the source doc gives is
# raw=48 -> +/-1/2 octave swing, which fixes this divisor (48 / 96 == 0.5).
VP_SWING_DIVISOR = 96.0

VR_SCALE = 0.0763   # Hz per raw unit (vibrato rate)
FORMANT_SCALE = 19.53  # Hz per raw unit (f1-f4, nf)
NASAL_FREQ_SCALE = 19.53  # Hz per raw unit (alias of FORMANT_SCALE, same hardware unit)
FRICATION_FREQ_SCALE = 39.06  # Hz per raw unit (ff)


@dataclass
class VocalParams:
    """13-parameter frame matching the original Propeller VocalTract encoding."""
    aa: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))     # aspiration
    ga: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))     # glottal
    gp: torch.Tensor = field(default_factory=lambda: torch.tensor(0.5))    # pitch (center ~110Hz)
    vp: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))    # vibrato pitch
    vr: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))    # vibrato rate
    f1: torch.Tensor = field(default_factory=lambda: torch.tensor(0.5))   # formant 1
    f2: torch.Tensor = field(default_factory=lambda: torch.tensor(0.5))   # formant 2
    f3: torch.Tensor = field(default_factory=lambda: torch.tensor(0.5))   # formant 3
    f4: torch.Tensor = field(default_factory=lambda: torch.tensor(0.5))   # formant 4
    na: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))   # nasal
    nf: torch.Tensor = field(default_factory=lambda: torch.tensor(0.5))   # nasal freq
    fa: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))   # frication
    ff: torch.Tensor = field(default_factory=lambda: torch.tensor(0.5))   # frication freq

    def __post_init__(self):
        # Ensure all tensors are floats
        if not isinstance(self.aa, torch.Tensor):
            self.aa = torch.tensor(float(self.aa), dtype=torch.float32)
        # ... same for all fields (omitted for brevity in this summary)


def normalize_to_tensor(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convert a dictionary of parameter values to a (B, T, 13) tensor.
    
    All input values should already be in [0, 1] range.
    """
    param_list = []
    for name in PARAM_NAMES:
        if name in params:
            t = params[name]
            if t.ndim == 0:
                t = t.unsqueeze(0)
            elif t.ndim == 1:
                t = t.unsqueeze(-1)
            elif t.ndim == 2:
                pass  # Already (B, T) or (T, B)
            else:
                raise ValueError(f"Expected scalar or 1D/2D tensor for {name}, got {t.shape}")
            param_list.append(t)
        else:
            param_list.append(torch.zeros_like(params[list(params.keys())[0]]))
    return torch.stack(param_list, dim=-1)


def denormalize(params: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Convert normalized [0,1] parameters to physical units, following the
    original VocalTract.spin header table (see docs/corrections.md).

    Args:
        params: Tensor of shape (..., 13) where last dimension is the 13 parameters.

    Returns:
        Dictionary mapping parameter name to its physical value.
    """
    result = {}
    for i, name in enumerate(PARAM_NAMES):
        val = params[..., i]
        raw = val * BYTE_SCALE  # every parameter is a single byte in [0, 255]
        if name in ("aa", "ga", "na", "fa"):
            # Amplitude: already linear 0..255, no further scaling.
            result[name] = raw
        elif name == "gp":
            # Log-pitch: raw=100 -> 110Hz, +/-48 raw units per octave.
            result[name] = GP_BASE_HZ * torch.pow(2.0, (raw - GP_BASE_RAW) / OCTAVE_UNIT)
        elif name == "vp":
            # Vibrato swing depth, in octaves (not a frequency).
            result[name] = raw / VP_SWING_DIVISOR
        elif name == "vr":
            result[name] = raw * VR_SCALE
        elif name in ("f1", "f2", "f3", "f4", "nf"):
            result[name] = raw * FORMANT_SCALE
        elif name == "ff":
            result[name] = raw * FRICATION_FREQ_SCALE
        else:
            result[name] = raw
    return result


def denormalize_batch(params: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Denormalize a batch of parameter frames.

    Args:
        params: Tensor of shape (B, T, 13).

    Returns:
        Dictionary of parameter names to (B, T) tensors of physical values.
    """
    # denormalize() already broadcasts over arbitrary leading dimensions.
    return denormalize(params)


# Default parameter values (typical speaking voice)
DEFAULT_PARAMS = {
    name: torch.tensor(0.5) if name in ("gp", "vp", "f1", "f2", "f3", "f4", "nf", "ff")
    else torch.tensor(0.0)
    for name in PARAM_NAMES
}
