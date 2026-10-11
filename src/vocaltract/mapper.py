"""
Parameter-to-LPC coefficient mapper.

This module implements a differentiable neural network that converts VocalTract
parameter frames (13 parameters) into LPC (Linear Prediction Coding) coefficients,
ensuring filter stability through reflection coefficients.

The pipeline:
1. Neural network: params (B,T,13) -> hidden representations -> reflection coefficients (B,T,order)
2. Levinson-Durbin recursion: reflection coefficients -> LPC coefficients (B,T,order)

All operations are differentiable for end-to-end training.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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