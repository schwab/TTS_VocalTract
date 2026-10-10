"""Unit tests for the Phase 1 placeholder excitation source."""

import numpy as np
import pytest
import torch

from vocaltract.params import PARAM_NAMES, denormalize
from vocaltract.source import make_excitation, sine_from_frequency, upsample_frames

SR = 16000
GP, GA, AA = (PARAM_NAMES.index(n) for n in ("gp", "ga", "aa"))


def constant_params(frames: int, gp: float, ga: float, aa: float) -> torch.Tensor:
    """(1, frames, 13) params with fixed source controls and zeros elsewhere."""
    params = torch.zeros(1, frames, len(PARAM_NAMES))
    params[..., GP], params[..., GA], params[..., AA] = gp, ga, aa
    return params


def peak_frequency(signal: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    return np.fft.rfftfreq(len(signal), 1 / SR)[spectrum.argmax()]


class TestShapes:
    @pytest.mark.parametrize("batch, frames, hop", [(2, 50, 320), (1, 1, 160), (3, 7, 64)])
    def test_output_shape(self, batch, frames, hop):
        torch.manual_seed(0)
        out = make_excitation(torch.rand(batch, frames, 13), sample_rate=SR, hop=hop)
        assert out.shape == (batch, frames * hop)

    def test_dtype_follows_params(self):
        out = make_excitation(torch.rand(1, 4, 13, dtype=torch.float64), hop=32)
        assert out.dtype == torch.float64

    @pytest.mark.parametrize("shape", [(50, 13), (1, 50, 12), (1, 1, 50, 13)])
    def test_bad_shape_raises(self, shape):
        with pytest.raises(ValueError):
            make_excitation(torch.rand(*shape))


class TestNumerics:
    def test_finite_for_random_inputs(self):
        torch.manual_seed(0)
        for _ in range(20):
            assert torch.isfinite(make_excitation(torch.rand(2, 50, 13))).all()

    def test_finite_at_range_extremes(self):
        for value in (0.0, 1.0):
            assert torch.isfinite(make_excitation(torch.full((1, 10, 13), value))).all()

    def test_pitch_matches_denormalized_gp(self):
        """The sine sits at denormalize(params)['gp'] Hz."""
        params = constant_params(50, gp=0.5, ga=1.0, aa=0.0)
        expected_hz = denormalize(params)["gp"][0, 0].item()
        out = make_excitation(params, sample_rate=SR, hop=320)[0].numpy()
        assert abs(peak_frequency(out) - expected_hz) < 2.0  # 1 Hz FFT bins

    def test_silent_when_amplitudes_zero(self):
        out = make_excitation(constant_params(10, gp=0.5, ga=0.0, aa=0.0))
        assert torch.count_nonzero(out) == 0

    def test_amplitude_scaling(self):
        """Voiced part peaks near ga; noise std tracks aa."""
        voiced = make_excitation(constant_params(50, gp=0.5, ga=0.3, aa=0.0))
        assert voiced.abs().max().item() == pytest.approx(0.3, rel=1e-3)

        torch.manual_seed(0)
        noisy = make_excitation(constant_params(50, gp=0.5, ga=0.0, aa=0.2))
        assert noisy.std().item() == pytest.approx(0.2, rel=0.05)

    def test_noise_reproducible_with_generator(self):
        params = torch.rand(1, 10, 13)
        a = make_excitation(params, generator=torch.Generator().manual_seed(1))
        b = make_excitation(params, generator=torch.Generator().manual_seed(1))
        assert torch.equal(a, b)


class TestPhaseContinuity:
    def test_no_jumps_when_pitch_changes(self):
        """A pitch sweep has no sample-to-sample jumps larger than the max frequency allows."""
        params = torch.zeros(1, 50, 13)
        params[..., GA] = 1.0
        params[..., GP] = torch.linspace(0.0, 1.0, 50)
        out = make_excitation(params, sample_rate=SR, hop=320)[0]

        max_hz = denormalize(params)["gp"].max().item()
        max_step = 2 * np.pi * max_hz / SR  # |d sin| <= |d phase|
        assert out.diff().abs().max().item() <= max_step * 1.01

    def test_sine_starts_at_zero_phase(self):
        sine = sine_from_frequency(torch.full((1, 100), 440.0), SR)
        assert sine[0, 0].item() == 0.0

    def test_long_signal_precision(self):
        """Phase wrapping keeps float32 accurate over a long signal."""
        n = 10 * SR
        f = 1000.0
        sine = sine_from_frequency(torch.full((1, n), f), SR)[0, -SR:].double()
        t = torch.arange(n - SR, n, dtype=torch.float64)
        ref = torch.sin(2 * torch.pi * f * t / SR)
        assert (sine - ref).abs().max().item() < 1e-3


class TestUpsample:
    def test_constant_frames_stay_constant(self):
        frames = torch.full((2, 5, 3), 0.7)
        assert torch.allclose(upsample_frames(frames, 16), torch.full((2, 80, 3), 0.7))

    def test_linear_between_frames(self):
        frames = torch.tensor([[[0.0], [1.0]]])
        up = upsample_frames(frames, 100)[0, :, 0]
        assert up.shape == (200,)
        assert torch.all(up.diff() >= 0)  # monotone ramp from first to second frame
        assert up[0].item() == 0.0 and up[-1].item() == 1.0


class TestGradients:
    def test_grads_flow_to_params(self):
        torch.manual_seed(0)
        params = torch.rand(2, 20, 13, requires_grad=True)
        make_excitation(params, hop=64).pow(2).mean().backward()
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()
        for idx in (GP, GA, AA):
            assert params.grad[..., idx].abs().sum() > 0, PARAM_NAMES[idx]

    def test_unused_params_get_zero_grad(self):
        """Phase 1 source only depends on gp, ga, aa."""
        torch.manual_seed(0)
        params = torch.rand(1, 10, 13, requires_grad=True)
        make_excitation(params, hop=64).sum().backward()
        unused = [i for i in range(13) if i not in (GP, GA, AA)]
        assert params.grad[..., unused].abs().sum() == 0

    def test_gradcheck(self):
        """Analytic gradients match finite differences (fixed-seed noise, so the output is deterministic)."""
        torch.manual_seed(0)
        params = torch.rand(1, 3, 13, dtype=torch.float64)
        params.requires_grad_()

        def fn(p):
            gen = torch.Generator().manual_seed(0)
            return make_excitation(p, hop=8, generator=gen)

        assert torch.autograd.gradcheck(fn, (params,))
