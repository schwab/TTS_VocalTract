"""
Parameter-to-LPC coefficient mapper.

Two interchangeable mappers convert VocalTract parameter frames (13 parameters)
into LPC coefficients:

- :class:`FormantToLPC` (default): cascades one resonator per formant parameter
  f1-f4, as VocalTract does. Formants follow the parameters by construction.
- :class:`ParamToLPC`: the original MLP mapper (all 13 parameters -> reflection
  coefficients). Kept for comparison; its parameters are not identifiable.

ParamToLPC details: a differentiable neural network that converts VocalTract
parameter frames (13 parameters) into LPC (Linear Prediction Coding) coefficients,
ensuring filter stability through reflection coefficients.

The pipeline:
1. Neural network: params (B,T,13) -> hidden representations -> reflection coefficients (B,T,order)
2. Levinson-Durbin recursion: reflection coefficients -> LPC coefficients (B,T,order)

All operations are differentiable for end-to-end training.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import BYTE_SCALE, FORMANT_SCALE, PARAM_NAMES

# Default bandwidth-expansion factor. Every pole radius is multiplied by GAMMA,
# so no pole can get closer to the unit circle than 1 - GAMMA. At 16 kHz,
# 0.99 means a minimum resonance bandwidth of about -ln(0.99) * 16000 / pi ~= 51 Hz,
# at the narrow end of natural formant bandwidths (~50-200 Hz).
DEFAULT_GAMMA = 0.99

# Reflection coefficients are capped at +/-MAX_REFLECTION. tanh rounds to exactly
# +/-1 for large inputs, and |k| = 1 everywhere makes the LPC polynomial's roots
# coincide (a 10-fold root): rounding the coefficients alone then moves real
# roots outside the unit circle, and no bandwidth expansion can fix that.
MAX_REFLECTION = 0.99


class ParamToLPC(nn.Module):
    """
    Convert VocalTrack parameter frames to LPC coefficients via reflection coefficients.

    Architecture:
        - Input: (batch, time, 13) parameter tensor
        - Two fully-connected layers with GELU activation
        - Output layer produces raw values in (-inf, +inf)
        - MAX_REFLECTION * tanh maps output into (-0.99, 0.99), interpreted as
          reflection coefficients (PARCOR)
        - Levinson-Durbin recursion converts PARCOR to LPC coefficients (float64)
        - Bandwidth expansion scales a_i by gamma**i, moving every pole radius
          r -> gamma * r
        - Output is float64: order-10 direct-form coefficients with poles near
          the unit circle are not stable once rounded to float32 (measured:
          float32 needs |k| <= 0.9 to keep every pole inside, float64 is
          exact at |k| = 0.99).

    Why bandwidth expansion?
        Reflection coefficients in (-1, 1) guarantee stability mathematically,
        but training can push them so close to +/-1 that poles sit within
        ~1e-10 of the unit circle (tanh even rounds to exactly 1 in float32).
        The filter's gain then overflows float32 and training diverges.
        Expansion keeps a fixed margin: every pole radius is <= gamma.

    Why reflection coefficients?
        - Guarantee pole stability by construction (all poles inside unit circle)
        - Well-trodden path in speech processing (used in GOLF codebase)
        - Numerically more stable than direct LPC optimization

    Args:
        order: Order of the LPC filter (number of reflection coefficients).
               Typically 8-16 for speech.
        hidden: Hidden dimension size for the two-layer MLP.
        gamma: Bandwidth-expansion factor in (0, 1]. 1.0 disables expansion.
    """

    def __init__(self, order: int = 10, hidden: int = 64, gamma: float = DEFAULT_GAMMA):
        super().__init__()
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.order = order
        self.gamma = gamma
        # gamma**i for i = 1..order; a buffer so it follows .to(device/dtype)
        self.register_buffer(
            "expansion", gamma ** torch.arange(1, order + 1, dtype=torch.float64), persistent=False
        )

        # Two-layer MLP: 13 -> hidden -> hidden -> order
        self.fc1 = nn.Linear(13, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, order)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert parameter frames to LPC coefficients.

        Inputs with fewer than 3 dimensions are promoted to (B, T, 13) by
        adding implicit leading batch/time dimensions of size 1:
        - (B, T, 13) -> (B, T, order)
        - (T, 13)    -> (1, T, order)      [implicit batch of 1]
        - (13,)      -> (1, 1, order)      [implicit batch of 1, time of 1]

        Args:
            params: Tensor where the last dimension is the 13 parameters.

        Returns:
            float64 tensor of shape (B, T, order) containing LPC coefficients.
        """
        if params.ndim == 1:
            params = params.unsqueeze(0).unsqueeze(0)
        elif params.ndim == 2:
            params = params.unsqueeze(0)
        elif params.ndim != 3:
            raise ValueError(f"Expected 1D, 2D, or 3D input, got {params.ndim}D")

        batch, time, _ = params.shape

        # Flatten batch/time dimensions for MLP processing
        params_flat = params.reshape(batch * time, 13)

        # Forward through MLP
        x = F.gelu(self.fc1(params_flat))
        x = F.gelu(self.fc2(x))
        raw_reflection = self.fc_out(x)  # Shape: (batch*time, order)

        # Map to reflection coefficients in (-1, 1)
        reflection = MAX_REFLECTION * torch.tanh(raw_reflection)  # Shape: (batch*time, order)

        # Convert reflection coefficients to LPC coefficients using Levinson-Durbin,
        # in float64 (see class docstring)
        lpc_coeffs = self._levinson_durbin(reflection.double())  # Shape: (batch*time, order)

        # Bandwidth expansion: pole radii r -> gamma * r
        lpc_coeffs = lpc_coeffs * self.expansion.double()

        # Reshape back to (B, T, order)
        return lpc_coeffs.view(batch, time, self.order)

    def _levinson_durbin(self, k: torch.Tensor) -> torch.Tensor:
        """
        Convert reflection coefficients (PARCOR) to LPC coefficients using Levinson-Durbin.

        This is implemented differentiably in pure PyTorch. We loop over the order
        dimension while keeping batch dimensions vectorized.

        The Levinson-Durbin recursion (for real-valued signals):
            For order m:
                A_m[j] = A_{m-1}[j] - k_m * A_{m-1}[m-j]   for j = 1..m-1
                A_m[m] = k_m

        Algorithm:
            A_1[k] = k_1  (first-order AR coeff = first reflection coeff)

            For m = 2..order:
                For j = 1..m-1:
                    A_m[j] = A_{m-1}[j] - k_m * A_{m-1}[m-j]
                A_m[m] = k_m

        Args:
            k: Reflection coefficients of shape (*, order) where last dim is the order.
               Each position k[..., i] is the (i+1)-th reflection coefficient k_{i+1}.

        Returns:
            LPC coefficients of shape (*, order) where last dim is the order.
            LPC coefficients are returned as [a1, a2, ..., a_order] where
            the filter polynomial is: 1 - sum(a_i * z^i)
        """
        # Input shape: (*, order) where * can be batch dimensions
        num_batch_dims = k.ndim - 1
        batch_shape = k.shape[:-1]
        order = k.shape[-1]

        # Create result tensor
        a = k.clone()  # Start with first-order: A_1 = k_1

        # Loop over orders 2..order
        for m in range(2, order + 1):
            # Get previous order coefficients and current reflection coefficient
            a_prev = a[..., :m-1]  # Shape: (*, m-1)
            k_m = k[..., m-1]    # Shape: (*)

            # Update: A_m[j] = A_{m-1}[j] - k_m * A_{m-1}[m-j]
            # We need to reverse a_prev to get A_{m-1}[m-j]
            a_prev_reversed = torch.flip(a_prev, dims=(-1,))

            # Element-wise multiplication and subtraction
            new_a = a_prev - k_m.unsqueeze(-1) * a_prev_reversed

            # Append k_m as the last coefficient
            a = torch.cat([new_a, k_m.unsqueeze(-1)], dim=-1)  # Shape: (*, m)

        return a

    def to_reflection(self, lpc_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Inverse of _levinson_durbin: convert LPC coefficients to reflection
        coefficients via the Schur / step-down recursion.

        At order p, the top reflection coefficient is simply A_p[p]. The
        lower-order coefficients are then recovered by solving the forward
        recursion's defining pair of equations for A_{p-1}:

            A_{p-1}[j] = (A_p[j] + k_p * A_p[p-j]) / (1 - k_p^2),  j = 1..p-1

        Args:
            lpc_coeffs: LPC coefficients of shape (*, order).

        Returns:
            Reflection coefficients of shape (*, order), where position
            [..., p-1] holds k_p (the order-p reflection coefficient).
        """
        order = lpc_coeffs.shape[-1]

        a = lpc_coeffs.clone()  # working copy; do not mutate the caller's tensor
        k = torch.zeros_like(a)

        for p in range(order, 0, -1):
            k_p = a[..., p - 1].clone()
            k[..., p - 1] = k_p

            if p == 1:
                break

            front = a[..., : p - 1]               # A_p[1..p-1]
            front_reversed = torch.flip(front, dims=(-1,))  # A_p[p-1..1]

            denom = 1.0 - k_p.pow(2)
            denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)

            # A_{p-1}[1..p-1]
            a[..., : p - 1] = (front + k_p.unsqueeze(-1) * front_reversed) / denom.unsqueeze(-1)

        return k


# --- Formant resonator mapper -------------------------------------------------

FORMANT_NAMES = ("f1", "f2", "f3", "f4")

# VocalTract damps every formant resonator by the same factor: pole radius
# ~0.984 at its 20 kHz sample rate, i.e. -ln(0.984) * 20000 / pi ~= 100 Hz.
VOCALTRACT_BANDWIDTH_HZ = 100.0

# Learned bandwidths never go below this. At 16 kHz, 40 Hz is a pole radius of
# exp(-pi * 40 / 16000) ~= 0.992, the same safety margin as ParamToLPC's gamma.
MIN_BANDWIDTH_HZ = 40.0

# Real "tilt" poles are capped at +/-MAX_TILT_POLE (a single real pole at 0.9
# is a ~20 dB low-frequency boost: plenty to shape the source's spectral slope).
MAX_TILT_POLE = 0.9


def _as_frames(params: torch.Tensor) -> torch.Tensor:
    """Promote (13,) or (T, 13) parameter input to (B, T, 13)."""
    if params.ndim == 1:
        return params.unsqueeze(0).unsqueeze(0)
    if params.ndim == 2:
        return params.unsqueeze(0)
    if params.ndim != 3:
        raise ValueError(f"Expected 1D, 2D, or 3D input, got {params.ndim}D")
    return params


def _poly_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Batched polynomial product along the last dim: (..., m), (..., n) -> (..., m + n - 1)."""
    n = q.shape[-1]
    return sum(F.pad(p * q[..., j : j + 1], (j, n - 1 - j)) for j in range(n))


class FormantToLPC(nn.Module):
    """
    Convert parameter frames to LPC coefficients by cascading formant resonators.

    This is the VocalTract model itself: the formant parameters ``f1``-``f4``
    each place one resonance (a conjugate pole pair) at that frequency, so a
    parameter frame determines the filter and moving ``f2`` moves exactly the
    second formant. Unlike :class:`ParamToLPC`, there is no network between
    the parameters and the filter to "average away" formant motion, and the
    other 9 parameters cannot influence the filter at all.

    VocalTract's resonator structure (``complex_resonators=True``, default)
    ----------------------------------------------------------------------
    VocalTract does not cascade real two-pole sections. Each formant is a
    *complex* one-pole resonator, ``s[n] = r e^{j theta} (s[n-1] + u[n])``,
    the four are cascaded on a complex signal, and the output is the real
    part (see ``reference/engine.py``). With ``Q(z) = prod_i (1 - r_i
    e^{j theta_i} z^-1)`` that is exactly the pole-zero filter

        H(z) = N(z) / D(z),   D(z) = Q(z) conj(Q)(z),
        N(z) = Re[ e^{j sum theta_i} * conj(Q)(z) ]   (order 4),

    i.e. the same poles as the all-pole cascade plus a numerator that
    changes the spectral slope between and above the formants. Without it,
    optimizing against VocalTract audio moved F2 of /a/ by ~150 Hz to
    compensate. :meth:`forward` returns the all-pole part D (as LPC
    coefficients) and :meth:`numerator` returns N; the pipeline applies N as
    a per-frame FIR before the all-pole filter. ``complex_resonators=False``
    gives the plain all-pole cascade (``numerator`` returns None).

    Each resonator is the second-order section

        1 - 2 r cos(theta) z^-1 + r^2 z^-2,   theta = 2 pi f / sample_rate,
        r = exp(-pi * bandwidth / sample_rate)

    and the full polynomial is the product of the four sections and
    ``tilt_poles`` real first-order sections ``1 - p z^-1``, giving order
    ``8 + tilt_poles`` (10 by default, same as ParamToLPC).

    Learned (the only trainable parameters):
        - one bandwidth per formant, initialized to VocalTract's ~100 Hz and
          kept above MIN_BANDWIDTH_HZ (stored as log(bandwidth - minimum), so
          Adam's step changes it by a roughly constant ratio);
        - the real tilt poles, ``MAX_TILT_POLE * tanh(raw)``, initialized to 0
          (no tilt). They let the filter match the target's overall spectral
          slope, which differs between VocalTract's glottal pulse and the
          Phase 1 sawtooth source.

    Stability holds for any parameter values: every pole radius is at most
    max(exp(-pi * MIN_BANDWIDTH_HZ / sample_rate), MAX_TILT_POLE) < 1. The
    product is computed in float64 (see :class:`ParamToLPC` on precision).
    Rounding still moves poles when several formants crowd together, because
    the direct-form polynomial is ill-conditioned for clustered roots. The
    worst case, all four formants at 0 Hz, moves the largest radius from 0.98
    to 0.997, which is still stable. The bundled targets never put F1 below
    150 Hz.

    Args:
        sample_rate: Audio sample rate in Hz.
        bandwidths_hz: Initial bandwidth per formant, each > MIN_BANDWIDTH_HZ.
        tilt_poles: Number of real tilt poles (0 disables).
        complex_resonators: Model VocalTract's complex resonators (adds the
            numerator above). False gives a plain all-pole cascade.
    """

    input_names = FORMANT_NAMES

    def __init__(
        self,
        sample_rate: int = 16000,
        bandwidths_hz: Tuple[float, ...] = (VOCALTRACT_BANDWIDTH_HZ,) * len(FORMANT_NAMES),
        tilt_poles: int = 2,
        complex_resonators: bool = True,
    ):
        super().__init__()
        if len(bandwidths_hz) != len(FORMANT_NAMES):
            raise ValueError(f"need {len(FORMANT_NAMES)} bandwidths, got {len(bandwidths_hz)}")
        if min(bandwidths_hz) <= MIN_BANDWIDTH_HZ:
            raise ValueError(f"bandwidths must exceed {MIN_BANDWIDTH_HZ} Hz, got {bandwidths_hz}")
        if tilt_poles < 0:
            raise ValueError(f"tilt_poles must be >= 0, got {tilt_poles}")
        self.sample_rate = sample_rate
        self.complex_resonators = complex_resonators
        self.order = 2 * len(FORMANT_NAMES) + tilt_poles
        excess = torch.tensor(bandwidths_hz, dtype=torch.float64) - MIN_BANDWIDTH_HZ
        self.log_bandwidth_excess = nn.Parameter(excess.log())
        self.raw_tilt = nn.Parameter(torch.zeros(tilt_poles, dtype=torch.float64))
        self.register_buffer(
            "formant_index", torch.tensor([PARAM_NAMES.index(n) for n in FORMANT_NAMES]),
            persistent=False,
        )

    def bandwidths_hz(self) -> torch.Tensor:
        """Current formant bandwidths in Hz, shape (4,)."""
        return MIN_BANDWIDTH_HZ + self.log_bandwidth_excess.exp()

    def tilt(self) -> torch.Tensor:
        """Current real tilt poles, shape (tilt_poles,)."""
        return MAX_TILT_POLE * torch.tanh(self.raw_tilt)

    def formants_hz(self, params: torch.Tensor) -> torch.Tensor:
        """Formant frequencies (..., 4) in Hz from (..., 13) parameter frames (float64)."""
        normalized = params.double().index_select(-1, self.formant_index)
        return normalized * (BYTE_SCALE * FORMANT_SCALE)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert parameter frames to LPC coefficients.

        Accepts (B, T, 13), (T, 13) or (13,) like :class:`ParamToLPC`.

        Returns:
            float64 tensor (B, T, order) in the ``1 - sum a_i z^-i`` convention.
        """
        theta, radius = self._poles(params)
        ones = torch.ones_like(theta[..., 0:1])

        poly = ones
        for i in range(len(FORMANT_NAMES)):
            section = torch.cat(
                [ones, -2 * radius[i] * torch.cos(theta[..., i : i + 1]), radius[i] ** 2 * ones],
                dim=-1,
            )
            poly = _poly_mul(poly, section)
        for pole in self.tilt():
            poly = _poly_mul(poly, torch.cat([ones, -pole * ones], dim=-1))

        # A(z) = 1 + sum c_i z^-i  ->  1 - sum a_i z^-i with a_i = -c_i
        return -poly[..., 1:]

    def numerator(self, params: torch.Tensor) -> Optional[torch.Tensor]:
        """
        FIR numerator N(z) of VocalTract's complex resonator cascade (see class docstring).

        Returns:
            float64 tensor (B, T, 5) of coefficients [n_0, ..., n_4] of
            ``N(z) = sum n_k z^-k``, or None when ``complex_resonators`` is False.
        """
        if not self.complex_resonators:
            return None
        theta, radius = self._poles(params)
        poly = torch.exp(1j * theta.sum(dim=-1, keepdim=True))  # e^{j sum theta}, (B, T, 1)
        one = torch.ones_like(poly)
        for i in range(len(FORMANT_NAMES)):
            # conj(Q) factor: 1 - r e^{-j theta} z^-1
            poly = _poly_mul(poly, torch.cat([one, -radius[i] * torch.exp(-1j * theta[..., i : i + 1])], -1))
        return poly.real

    def _poles(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pole angles (B, T, 4) and radii (4,) for (..., 13) parameter frames."""
        theta = 2 * torch.pi * self.formants_hz(_as_frames(params)) / self.sample_rate
        radius = torch.exp(-torch.pi * self.bandwidths_hz() / self.sample_rate)
        return theta, radius


def make_mapper(kind: str = "formant", sample_rate: int = 16000) -> nn.Module:
    """Build a parameter -> LPC mapper: ``"formant"`` (FormantToLPC) or ``"mlp"`` (ParamToLPC)."""
    if kind == "formant":
        return FormantToLPC(sample_rate=sample_rate)
    if kind == "mlp":
        return ParamToLPC()
    raise ValueError(f"unknown mapper {kind!r}, expected 'formant' or 'mlp'")


if __name__ == "__main__":
    # Quick demo
    print("Testing ParamToLPC...")
    model = ParamToLPC(order=10, hidden=64)

    # Test various shapes
    for shape in [(2, 50, 13), (50, 13), (13,)]:
        params = torch.rand(*shape)
        lpc = model(params)
        print(f"Input shape: {params.shape} -> Output shape: {lpc.shape}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    params = torch.rand(2, 50, 13, requires_grad=True)
    lpc = model(params)
    loss = lpc.pow(2).sum()
    loss.backward()
    print(f"Gradient norm on params: {params.grad.abs().sum().item():.6f}")

    # Test stability
    print("\nTesting pole stability...")
    torch.manual_seed(42)
    params = torch.rand(5, 10, 13)
    lpc = model(params)

    import numpy as np
    all_stable = True
    for b in range(5):
        for t in range(10):
            coeffs = lpc[b, t].cpu().numpy()
            # Build polynomial: 1 - a1*z - a2*z^2 - ... - a_n*z^n
            coeffs_np = -coeffs  # Flip sign for polynomial
            coeffs_np = np.append(1, coeffs_np)
            roots = np.roots(coeffs_np)
            pole_radii = np.abs(roots)
            if np.any(pole_radii >= 1.0 - 1e-6):
                print(f"UNSTABLE! Filter ({b},{t}): max pole radius = {np.max(pole_radii)}")
                all_stable = False

    if all_stable:
        print("All filters are stable!")