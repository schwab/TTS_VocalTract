"""Tests for the differentiable resample_poly."""

import numpy as np
import pytest
import scipy.signal
import torch

from vocaltract.resample import resample_poly


@pytest.mark.parametrize("up,down", [(4, 5), (5, 4), (1, 2), (3, 1), (8, 10)])
def test_matches_scipy(up, down):
    x = np.random.default_rng(0).standard_normal((2, 2001))
    ref = scipy.signal.resample_poly(x, up, down, axis=-1)
    ours = resample_poly(torch.tensor(x), up, down).numpy()
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_identity():
    x = torch.randn(1, 100)
    assert resample_poly(x, 3, 3) is x


def test_keeps_leading_dims_and_dtype():
    out = resample_poly(torch.randn(2, 3, 500), 4, 5)
    assert out.shape == (2, 3, 400) and out.dtype == torch.float32


def test_gradient_flows():
    x = torch.randn(1, 500, dtype=torch.float64, requires_grad=True)
    resample_poly(x, 4, 5).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_removes_content_above_new_nyquist():
    """A 9 kHz tone at 20 kHz is attenuated ~30 dB on conversion to 16 kHz (scipy's Kaiser beta 5)."""
    t = torch.arange(20000, dtype=torch.float64) / 20000
    tone = torch.sin(2 * torch.pi * 9000 * t)[None]
    assert resample_poly(tone, 4, 5)[0, 200:-200].abs().max() < 0.05
