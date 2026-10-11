"""Tests for the shared synthesis pipeline and filter gain normalization."""

import numpy as np
import pytest
import torch
from scipy.signal import freqz

from vocaltract.filter import LPCFilter, fir_frames, frames_to_samples, min_response, peak_gain
from vocaltract.params import PARAM_NAMES
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


class TestComplexResonators:
    """FormantToLPC's numerator reproduces VocalTract's complex resonator cascade."""

    def test_numerator_shape_and_flag(self):
        from vocaltract.mapper import FormantToLPC

        params = torch.rand(2, 7, 13)
        assert FormantToLPC()(params).shape == (2, 7, 10)
        assert FormantToLPC().numerator(params).shape == (2, 7, 5)
        assert FormantToLPC(complex_resonators=False).numerator(params) is None

    def test_pole_zero_equals_complex_cascade(self):
        """FIR numerator + all-pole filter == four complex one-pole resonators, real part."""
        from vocaltract.mapper import FormantToLPC

        mapper = FormantToLPC(tilt_poles=0)
        params = torch.rand(1, 1, 13, dtype=torch.float64)
        theta = (2 * np.pi * mapper.formants_hz(params) / mapper.sample_rate)[0, 0].numpy()
        radius = mapper.bandwidths_hz().detach().mul(-np.pi / mapper.sample_rate).exp().numpy()

        x = np.random.default_rng(0).standard_normal(3000)
        signal = x.astype(complex)
        for t, r in zip(theta, radius):
            state, out = 0j, np.empty_like(signal)
            for n, u in enumerate(signal):
                state = r * np.exp(1j * t) * (state + u)
                out[n] = state
            signal = out
        reference = signal.real / np.prod(radius)

        n = len(x)
        y = LPCFilter()(
            fir_frames(torch.tensor(x)[None], mapper.numerator(params).detach().expand(1, n, 5)),
            mapper(params).detach().expand(1, n, 8),
        )[0].numpy()
        np.testing.assert_allclose(y, reference, atol=1e-8 * np.abs(reference).max())

    def test_min_response_with_numerator_matches_scipy(self):
        from vocaltract.mapper import FormantToLPC

        mapper = FormantToLPC()
        params = torch.rand(1, 1, 13, dtype=torch.float64)
        a, b = mapper(params).detach()[0, 0], mapper.numerator(params).detach()[0, 0]
        _, h = freqz(b.numpy(), np.concatenate([[1.0], -a.numpy()]), worN=8192)
        expected = 1 / np.abs(h).max()
        assert min_response(a, n_fft=16384, numerator=b).item() == pytest.approx(expected, rel=1e-3)

    def test_trace_records_numerator(self):
        from vocaltract.mapper import FormantToLPC

        trace = synthesize_trace(torch.rand(1, 4, 13), FormantToLPC(), hop=32)
        assert trace.frame_numerator.shape == (1, 4, 5)
        assert torch.isfinite(trace.audio).all()


class TestRenderRate:
    def test_output_at_sample_rate(self):
        from vocaltract.mapper import FormantToLPC

        params = torch.rand(1, 5, 13)
        trace = synthesize_trace(params, FormantToLPC(sample_rate=20000), hop=320,
                                 render_rate=20000, waveform="vocaltract_aliased")
        assert trace.audio.shape == (1, 1600) and trace.audio.dtype == torch.float32
        assert trace.excitation.shape == (1, 2000)  # chain runs at the render rate

    def test_matches_scipy_resample_of_native_render(self):
        from scipy.signal import resample_poly as scipy_resample

        from vocaltract.mapper import FormantToLPC

        params = torch.rand(1, 5, 13, dtype=torch.float64)
        mapper = FormantToLPC(sample_rate=20000)
        gen = lambda: torch.Generator().manual_seed(0)  # noqa: E731
        with torch.no_grad():
            native = synthesize(params, mapper, hop=400, sample_rate=20000, generator=gen())
            ours = synthesize(params, mapper, hop=320, render_rate=20000, generator=gen())
        np.testing.assert_allclose(ours[0].numpy(), scipy_resample(native[0].numpy(), 4, 5),
                                   atol=1e-10)

    def test_mapper_rate_must_match(self):
        from vocaltract.mapper import FormantToLPC

        with pytest.raises(ValueError, match="mapper"):
            synthesize(torch.rand(1, 2, 13), FormantToLPC(sample_rate=16000), render_rate=20000)

    def test_hop_must_divide(self):
        from vocaltract.mapper import FormantToLPC

        with pytest.raises(ValueError, match="hop"):
            synthesize(torch.rand(1, 2, 13), FormantToLPC(sample_rate=20000), hop=321,
                       render_rate=20000)

    def test_gradients_flow(self):
        from vocaltract.mapper import FormantToLPC

        params = torch.rand(1, 4, 13, requires_grad=True)
        synthesize(params, FormantToLPC(sample_rate=20000), render_rate=20000,
                   waveform="vocaltract_aliased").pow(2).sum().backward()
        for name in ("gp", "ga", "f1", "f4"):
            assert params.grad[..., PARAM_NAMES.index(name)].abs().sum() > 0, name
