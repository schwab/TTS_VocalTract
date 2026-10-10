"""Unit tests for the ParamToLPC mapper."""

import numpy as np
import pytest
import torch

from vocaltract.mapper import ParamToLPC


class TestParamToLPCShape:
    """Test output shapes."""

    def test_basic_shapes(self):
        """Test various input shapes produce correct output shapes."""
        model = ParamToLPC(order=10, hidden=64)

        # (B, T, 13) -> (B, T, order)
        params = torch.rand(2, 50, 13)
        lpc = model(params)
        assert lpc.shape == (2, 50, 10)

        # (T, 13) -> (1, T, order) - implicit batch of 1
        params_1d = torch.rand(50, 13)
        lpc_1d = model(params_1d)
        assert lpc_1d.shape == (1, 50, 10)

        # Scalar -> (1, 1, order)
        params_scalar = torch.rand(13)
        lpc_scalar = model(params_scalar)
        assert lpc_scalar.shape == (1, 1, 10)


class TestReflectionStability:
    """Test that reflection coefficients stay in (-1, 1)."""

    @pytest.mark.parametrize("num_samples", [10, 100, 1000])
    def test_reflection_bounds(self, num_samples):
        """All reflection coefficients should be in (-1, 1)."""
        torch.manual_seed(42)
        model = ParamToLPC()
        
        for _ in range(num_samples):
            params = torch.rand(1, 1, 13)
            lpc = model(params)
            
            # Check that lpc values are in valid range for reflection coefficients
            assert lpc.min() > -1.0, f"Reflection coefficient {lpc.min()} is <= -1"
            assert lpc.max() < 1.0, f"Reflection coefficient {lpc.max()} is >= 1"


class TestPoleStability:
    """Test that LPC coefficients produce stable filters (poles inside unit circle)."""

    def test_pole_stability(self):
        """All poles should lie strictly inside the unit circle."""
        torch.manual_seed(123)
        model = ParamToLPC(order=10)

        num_trials = 100
        for _ in range(num_trials):
            params = torch.rand(1, 1, 13)
            lpc = model(params)

            # LPC polynomial: z^n - a[0]*z^(n-1) - ... - a[n-1] = 0
            # (torch has no polyroots; numpy.roots is the standard tool here,
            # same as the sanity check in mapper.py's own __main__ demo.)
            coeffs = lpc.flatten().detach().numpy()
            poly = np.concatenate([[1.0], -coeffs])
            roots = np.roots(poly)

            max_mag = np.abs(roots).max()
            assert max_mag < 1.0, f"Unstable pole found: max |root| = {max_mag}"


class TestGradientFlow:
    """Test that gradients flow through the mapper."""

    def test_gradient_flows(self):
        """Gradients should propagate back through the mapper."""
        torch.manual_seed(42)
        model = ParamToLPC(order=10)
        
        params = torch.rand(2, 50, 13, requires_grad=True)
        lpc = model(params)
        
        loss = lpc.pow(2).sum()
        loss.backward()
        
        assert params.grad is not None, "No gradient on input params"
        assert params.grad.numel() > 0, "Gradient has zero elements"
        assert params.grad.abs().sum().item() > 0, "All gradients are zero"
        
        # Also check model parameters have gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.abs().sum().item() > 0, f"No gradient on {name}"


class TestDeterminism:
    """Test reproducibility with fixed seed."""

    def test_deterministic_output(self):
        """Same input should produce same output."""
        torch.manual_seed(999)
        model1 = ParamToLPC(order=10)
        
        torch.manual_seed(999)
        model2 = ParamToLPC(order=10)
        
        params = torch.rand(3, 100, 13)
        
        lpc1 = model1(params)
        lpc2 = model2(params)
        
        assert torch.allclose(lpc1, lpc2), "Models produce different outputs"


class TestInverseTransform:
    """Test that to_reflection correctly inverts _levinson_durbin."""

    def test_inverse_consistency(self):
        """k -> LPC (via _levinson_durbin) -> k (via to_reflection) should round-trip.

        Note: this is a roundtrip over the *reflection coefficients* consumed by
        _levinson_durbin, not over the original 13-dim params (the mapper's output
        has `order` dims, which need not equal 13, so comparing against `params`
        directly is a shape mismatch, not a meaningful identity).
        """
        torch.manual_seed(42)
        model = ParamToLPC(order=10)

        k = torch.tanh(torch.randn(2, 3, model.order))

        lpc = model._levinson_durbin(k)
        recovered_k = model.to_reflection(lpc)

        # float32 recursion drift accumulates over `order` steps; 1e-3 comfortably
        # separates that from a real algorithmic bug (which shows up as O(1) error).
        assert torch.allclose(recovered_k, k, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
