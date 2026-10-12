"""Unit tests for the Phase 1 placeholder excitation source."""

import numpy as np
import pytest
import torch

from vocaltract.params import PARAM_NAMES, denormalize
from vocaltract.source import (
    harmonics_from_frequency,
    make_excitation,
    sine_from_frequency,
    upsample_frames,
)

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
        sine = make_excitation(constant_params(50, gp=0.5, ga=0.3, aa=0.0), max_harmonics=1)
        assert sine.abs().max().item() == pytest.approx(0.3, rel=1e-3)
        harmonic = make_excitation(constant_params(50, gp=0.5, ga=0.3, aa=0.0))
        assert 0.25 < harmonic.abs().max().item() < 0.36  # sawtooth ~1 plus Gibbs overshoot

        torch.manual_seed(0)
        noisy = make_excitation(constant_params(50, gp=0.5, ga=0.0, aa=0.2))
        assert noisy.std().item() == pytest.approx(0.2, rel=0.05)

    def test_noise_reproducible_with_generator(self):
        params = torch.rand(1, 10, 13)
        a = make_excitation(params, generator=torch.Generator().manual_seed(1))
        b = make_excitation(params, generator=torch.Generator().manual_seed(1))
        assert torch.equal(a, b)


class TestHarmonics:
    def test_harmonics_present_with_inverse_k_amplitudes(self):
        f0 = 200.0
        out = harmonics_from_frequency(torch.full((1, SR), f0), SR)[0].numpy()
        spectrum = np.abs(np.fft.rfft(out))  # 1 Hz bins, integer-period signal: no window needed
        fundamental = spectrum[int(f0)]
        for k in (2, 3, 5, 10, 37):  # 37 * 200 = 7400 Hz, below the Nyquist taper
            assert spectrum[int(k * f0)] / fundamental == pytest.approx(1 / k, rel=1e-3), k

    def test_no_energy_at_or_above_nyquist_or_between_harmonics(self):
        """f0 = 3000 Hz: only 3000 and 6000 Hz fit below 8 kHz; 9000 Hz would alias to 7000."""
        out = harmonics_from_frequency(torch.full((1, SR), 3000.0), SR)[0].numpy()
        spectrum = np.abs(np.fft.rfft(out))
        strong = set(np.flatnonzero(spectrum > 1e-3 * spectrum.max()))
        assert strong == {3000, 6000}  # 6000 Hz is 2 kHz below Nyquist: full weight

    def test_continuous_as_harmonic_crosses_nyquist(self):
        """The Nyquist taper fades harmonics out, so output changes smoothly with pitch."""
        n = 64  # short, so phase drift between the two pitches is negligible
        f_cross = 8000.0 / 5  # 5th harmonic sits exactly at Nyquist
        below = harmonics_from_frequency(torch.full((1, n), f_cross - 1e-3, dtype=torch.float64), SR)
        above = harmonics_from_frequency(torch.full((1, n), f_cross + 1e-3, dtype=torch.float64), SR)
        # A hard cutoff would jump by the 5th harmonic's full amplitude (2/pi/5 ~= 0.13).
        assert (below - above).abs().max().item() < 1e-3

    def test_harmonic_count_follows_lowest_pitch(self):
        """A pitch sweep keeps every harmonic below Nyquist at every sample."""
        f = torch.linspace(60.0, 900.0, SR)[None]
        out = harmonics_from_frequency(f, SR)
        assert torch.isfinite(out).all()

    def test_max_harmonics_one_is_pure_sine(self):
        f = torch.full((2, 1000), 330.0)
        assert torch.allclose(harmonics_from_frequency(f, SR, max_harmonics=1), sine_from_frequency(f, SR))

    def test_excitation_pitch_is_fundamental(self):
        """The fundamental is still the strongest component."""
        params = constant_params(50, gp=0.5, ga=1.0, aa=0.0)
        out = make_excitation(params, sample_rate=SR, hop=320)[0].numpy()
        assert abs(peak_frequency(out) - denormalize(params)["gp"][0, 0].item()) < 2.0


class TestPhaseContinuity:
    def test_no_jumps_when_pitch_changes(self):
        """A pitch sweep has no sample-to-sample jumps larger than the max frequency allows."""
        params = torch.zeros(1, 50, 13)
        params[..., GA] = 1.0
        params[..., GP] = torch.linspace(0.0, 1.0, 50)
        out = make_excitation(params, sample_rate=SR, hop=320, max_harmonics=1)[0]

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


class TestVocalTractWaveform:
    """VocalTract's glottal pulse g(phi) = -cos(2 pi 2**(16 (phi - 1)))."""

    def phases(self, f0: float = 100.0, n: int = 16000) -> np.ndarray:
        return (np.arange(n) * f0 / SR) % 1.0

    def test_band_limited_matches_closed_form_without_dc(self):
        from vocaltract.source import vocaltract_pulse

        freq = torch.full((1, SR), 100.0, dtype=torch.float64)
        ours = harmonics_from_frequency(freq, SR, waveform="vocaltract")[0].numpy()
        ref = vocaltract_pulse(self.phases())
        ref -= ref.mean()
        # Harmonics above Nyquist are dropped/tapered, so allow a small residual.
        assert np.abs(ours - ref).max() < 5e-3

    def test_aliased_is_the_closed_form(self):
        from vocaltract.source import vocaltract_pulse

        freq = torch.full((1, SR), 100.0, dtype=torch.float64)
        ours = harmonics_from_frequency(freq, SR, waveform="vocaltract_aliased")[0].numpy()
        np.testing.assert_allclose(ours, vocaltract_pulse(self.phases()), atol=1e-9)

    def test_pulse_shape(self):
        """Flat at -1 for most of the period, one pulse reaching +1 near the end."""
        from vocaltract.source import vocaltract_pulse

        phase = np.linspace(0, 1, 10001)[:-1]
        g = vocaltract_pulse(phase)
        assert np.all(np.abs(g[phase < 0.5] + 1) < 1e-3)
        assert phase[np.argmax(g)] == pytest.approx(1 - 1 / 16, abs=1e-3)

    def test_coefficients_reconstruct_pulse(self):
        from vocaltract.source import harmonic_coefficients, vocaltract_pulse

        cos_k, sin_k = (c.double().numpy() for c in harmonic_coefficients("vocaltract", 2000))
        phase = np.linspace(0, 1, 512, endpoint=False)
        k = np.arange(1, 2001)[:, None]
        series = cos_k @ np.cos(2 * np.pi * k * phase) + sin_k @ np.sin(2 * np.pi * k * phase)
        ref = vocaltract_pulse(phase)
        np.testing.assert_allclose(series, ref - ref.mean(), atol=1e-6)

    def test_sawtooth_coefficients(self):
        from vocaltract.source import SAWTOOTH_NORM, harmonic_coefficients

        cos_k, sin_k = harmonic_coefficients("sawtooth", 5)
        assert torch.all(cos_k == 0)
        assert torch.allclose(sin_k, SAWTOOTH_NORM / torch.arange(1.0, 6.0))

    @pytest.mark.parametrize("waveform", ["vocaltract", "vocaltract_aliased"])
    def test_excitation_gradients(self, waveform):
        params = constant_params(5, gp=0.4, ga=0.5, aa=0.0).requires_grad_()
        make_excitation(params, hop=64, waveform=waveform).pow(2).sum().backward()
        assert params.grad[..., PARAM_NAMES.index("gp")].abs().sum() > 0
        assert params.grad[..., PARAM_NAMES.index("ga")].abs().sum() > 0

    def test_unknown_waveform(self):
        with pytest.raises(ValueError):
            harmonics_from_frequency(torch.full((1, 10), 100.0), SR, waveform="square")


class TestEngineNoiseLevel:
    """Aspiration at the engine's level: uniform noise at a quarter of the pulse's peak."""

    def aspiration(self, waveform):
        from vocaltract.params import PARAM_NAMES

        params = torch.zeros(1, 50, 13)
        params[..., PARAM_NAMES.index("aa")] = 0.5
        params[..., PARAM_NAMES.index("gp")] = 0.4
        return make_excitation(params, hop=320, waveform=waveform,
                               generator=torch.Generator().manual_seed(0))

    @pytest.mark.parametrize("waveform", ["vocaltract", "vocaltract_aliased"])
    def test_vocaltract_waveforms_use_engine_level(self, waveform):
        from vocaltract.source import ENGINE_NOISE_RMS

        assert ENGINE_NOISE_RMS == pytest.approx(0.25 / 3 ** 0.5)
        assert self.aspiration(waveform).std().item() == pytest.approx(0.5 * ENGINE_NOISE_RMS, rel=0.02)

    def test_sawtooth_keeps_unit_variance_noise(self):
        assert self.aspiration("sawtooth").std().item() == pytest.approx(0.5, rel=0.02)
