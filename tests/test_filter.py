"""Unit tests for the LPCFilter wrapper."""

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from vocaltract.filter import HAS_TORCHLPC, LPCFilter, naive_sample_wise_lpc
from vocaltract.mapper import ParamToLPC


class TestShapes:
    """Output shape and dtype."""

    def test_spec_example_shape(self):
        """The Phase 1 spec example returns (2, 16000)."""
        torch.manual_seed(0)
        y = LPCFilter()(torch.randn(2, 16000), torch.randn(2, 16000, 10) * 0.01)
        assert y.shape == (2, 16000)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        torch.manual_seed(0)
        x = torch.randn(3, 500, dtype=dtype)
        a = torch.randn(3, 500, 8, dtype=dtype) * 0.01
        y = LPCFilter()(x, a)
        assert y.shape == x.shape
        assert y.dtype == dtype

    @pytest.mark.parametrize(
        "x_shape, a_shape",
        [((2, 100), (2, 99, 10)), ((100,), (1, 100, 10)), ((2, 100), (2, 100))],
    )
    def test_bad_shapes_raise(self, x_shape, a_shape):
        with pytest.raises(ValueError):
            LPCFilter()(torch.randn(*x_shape), torch.randn(*a_shape))


class TestFilterBehaviour:
    """Numerical correctness."""

    def test_near_zero_coeffs_is_identity(self):
        """With tiny coefficients the all-pole filter is approximately a pass-through."""
        torch.manual_seed(0)
        x = torch.randn(2, 4000)
        y = LPCFilter()(x, torch.randn(2, 4000, 10) * 1e-5)
        assert torch.allclose(y, x, atol=1e-3)

    def test_zero_coeffs_is_exact_identity(self):
        x = torch.randn(2, 1000)
        y = LPCFilter()(x, torch.zeros(2, 1000, 10))
        assert torch.allclose(y, x)

    def test_matches_scipy_lfilter(self):
        """Time-invariant coefficients match scipy's 1 / (1 - sum a_i z^-i)."""
        torch.manual_seed(0)
        order, T = 6, 2000
        model = ParamToLPC(order=order)
        a = model(torch.rand(1, 1, 13)).detach().double().squeeze()  # (order,)
        x = torch.randn(1, T, dtype=torch.float64)

        y = LPCFilter()(x, a.expand(1, T, order))
        ref = lfilter([1.0], np.concatenate([[1.0], -a.numpy()]), x.numpy()[0])
        np.testing.assert_allclose(y.numpy()[0], ref, rtol=1e-8, atol=1e-8)

    def test_matches_naive_reference(self):
        """Time-varying coefficients match the pure-PyTorch fallback loop."""
        torch.manual_seed(0)
        x = torch.randn(2, 300, dtype=torch.float64)
        a = ParamToLPC(order=8)(torch.rand(2, 300, 13)).detach().double()
        np.testing.assert_allclose(
            LPCFilter()(x, a).numpy(), naive_sample_wise_lpc(x, a).numpy(), rtol=1e-8, atol=1e-8
        )

    def test_mapper_output_is_stable(self):
        """Mapper coefficients produce bounded output over a long signal."""
        torch.manual_seed(0)
        x = torch.randn(4, 16000)
        a = ParamToLPC()(torch.rand(4, 16000, 13)).detach()
        y = LPCFilter()(x, a)
        assert torch.isfinite(y).all()


class TestGradients:
    """Gradient flow through the filter."""

    def test_grads_reach_excitation_and_coeffs(self):
        torch.manual_seed(0)
        x = torch.randn(2, 800, requires_grad=True)
        a = (torch.randn(2, 800, 10) * 0.05).requires_grad_()
        LPCFilter()(x, a).pow(2).mean().backward()
        for t in (x, a):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0

    def test_gradcheck(self):
        """Analytic (reverse-filter) gradients match finite differences."""
        torch.manual_seed(0)
        x = torch.randn(1, 40, dtype=torch.float64, requires_grad=True)
        a = (torch.randn(1, 40, 3, dtype=torch.float64) * 0.1).requires_grad_()
        assert torch.autograd.gradcheck(LPCFilter(), (x, a))

    def test_grads_reach_mapper_params(self):
        """End-to-end: params -> mapper -> filter -> loss populates params.grad."""
        torch.manual_seed(0)
        params = torch.rand(1, 200, 13, requires_grad=True)
        a = ParamToLPC()(params)
        LPCFilter()(torch.randn(1, 200), a).pow(2).mean().backward()
        assert params.grad is not None and params.grad.abs().sum() > 0


class TestFallback:
    """Naive fallback path behind the feature flag."""

    def test_fallback_flag(self):
        assert LPCFilter(use_torchlpc=False).use_torchlpc is False
        assert LPCFilter().use_torchlpc is HAS_TORCHLPC

    def test_fallback_matches(self):
        torch.manual_seed(0)
        x = torch.randn(2, 200, dtype=torch.float64)
        a = torch.randn(2, 200, 4, dtype=torch.float64) * 0.1
        np.testing.assert_allclose(
            LPCFilter(use_torchlpc=False)(x, a).numpy(),
            LPCFilter()(x, a).numpy(),
            rtol=1e-8,
            atol=1e-8,
        )


class TestAnalyze:
    """Reference LPC analysis helper."""

    def test_recovers_known_filter(self):
        """Analysis of white noise filtered by a known AR model recovers its coefficients."""
        torch.manual_seed(0)
        order, T = 4, 64000
        true_a = ParamToLPC(order=order)(torch.rand(1, 1, 13)).detach().squeeze()
        x = torch.randn(1, T, dtype=torch.float64)
        y = LPCFilter()(x, true_a.double().expand(1, T, order))

        est = LPCFilter.analyze(y, order)
        assert est.shape == (1, 1, order)
        np.testing.assert_allclose(est.squeeze().numpy(), true_a.double().numpy(), atol=0.03)

    def test_framed_shape_and_no_grad(self):
        audio = torch.randn(2, 1000, requires_grad=True)
        est = LPCFilter.analyze(audio, order=10, frame_length=250)
        assert est.shape == (2, 4, 10)
        assert not est.requires_grad
