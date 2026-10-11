"""Tests for the shared synthesis pipeline and filter gain normalization."""

import numpy as np
import pytest
import torch
from scipy.signal import freqz

from vocaltract.filter import LPCFilter, frames_to_samples, min_response, peak_gain
from vocaltract.mapper import ParamToLPC
from vocaltract.pipeline import synthesize, synthesize_trace
from vocaltract.source import make_excitation


def saturated_mapper(gamma: float = 0.99) -> ParamToLPC:
    """A mapper whose reflection coefficients are all exactly +1 before expansion."""
    mapper = ParamToLPC(gamma=gamma).double()
    with torch.no_grad():
        mapper.fc_out.weight.zero_()
        mapper.fc_out.bias.fill_(50.0)  # tanh(50) == 1.0 exactly
    return mapper


class TestGainFunctions:
    def test_min_response_matches_scipy(self):
        torch.manual_seed(0)
        a = ParamToLPC()(torch.rand(1, 4, 13)).detach().double()[0]  # (4, order)
        for row in a:
            _, h = freqz([1.0], np.concatenate([[1.0], -row.numpy()]), worN=4096)
            assert peak_gain(row, n_fft=8192).item() == pytest.approx(np.abs(h).max(), rel=1e-3)
            assert min_response(row, n_fft=8192).item() == pytest.approx(1 / np.abs(h).max(), rel=1e-3)

    def test_zero_coeffs_unit_gain(self):
        assert torch.allclose(peak_gain(torch.zeros(3, 10)), torch.ones(3))

    def test_gradient_finite_for_sharp_resonances(self):
        """min_response (not 1/peak_gain) keeps gradients bounded near the unit circle."""
        mapper = saturated_mapper()
        params = torch.rand(1, 3, 13, dtype=torch.float64, requires_grad=True)
        min_response(mapper(params)).sum().backward()
        assert torch.isfinite(params.grad).all()


class TestSynthesize:
    def test_audio_dtype_follows_params(self):
        torch.manual_seed(0)
        assert synthesize(torch.rand(1, 4, 13), ParamToLPC(), hop=32).dtype == torch.float32

    def test_trace_shapes(self):
        torch.manual_seed(0)
        trace = synthesize_trace(torch.rand(2, 10, 13), ParamToLPC(), hop=64)
        assert trace.frame_coeffs.shape == (2, 10, 10)
        assert trace.frame_norm.shape == (2, 10)
        assert trace.sample_coeffs.shape == (2, 640, 10)
        assert trace.excitation.shape == trace.audio.shape == (2, 640)

    def test_matches_manual_chain(self):
        torch.manual_seed(0)
        params, mapper = torch.rand(1, 8, 13), ParamToLPC()
        gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
        coeffs = mapper(params)
        excitation = make_excitation(params, hop=32, generator=gen())
        excitation = excitation * frames_to_samples(min_response(coeffs)[..., None], 32)[..., 0]
        manual = LPCFilter()(excitation, frames_to_samples(coeffs, 32)).float()
        assert torch.allclose(synthesize(params, mapper, hop=32, generator=gen()), manual)

    def test_normalize_gain_flag(self):
        torch.manual_seed(0)
        params, mapper = torch.rand(1, 8, 13), ParamToLPC()
        gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
        raw = synthesize(params, mapper, hop=32, generator=gen(), normalize_gain=False)
        manual = LPCFilter()(make_excitation(params, hop=32, generator=gen()),
                             frames_to_samples(mapper(params), 32)).float()
        assert torch.allclose(raw, manual)

    def test_normalization_bounds_output_for_sharp_filter(self):
        """A saturated mapper has a huge peak gain; normalization keeps the output level sane."""
        mapper = saturated_mapper()
        params = torch.rand(1, 20, 13, dtype=torch.float64)
        params[..., 0] = 0.0  # no aspiration noise
        params[..., 1] = 1.0  # full voicing
        trace = synthesize_trace(params, mapper, hop=160)
        raw = synthesize(params, mapper, hop=160, normalize_gain=False)
        assert (1 / trace.frame_norm).min() > 1e3  # the filter really is sharp
        assert trace.audio.abs().max() < 10
        assert raw.abs().max() > 100 * trace.audio.abs().max()
