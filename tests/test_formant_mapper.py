"""Unit tests for the FormantToLPC resonator mapper."""

import numpy as np
import pytest
import torch
from scipy.signal import freqz

from vocaltract.filter import LPCFilter
from vocaltract.mapper import (
    FORMANT_NAMES,
    MAX_TILT_POLE,
    MIN_BANDWIDTH_HZ,
    FormantToLPC,
    ParamToLPC,
    make_mapper,
)
from vocaltract.params import BYTE_SCALE, FORMANT_SCALE, PARAM_NAMES

SR = 16000
VOWEL_AH = (730.0, 1090.0, 2440.0, 3400.0)
VOWEL_EE = (270.0, 2290.0, 3010.0, 3500.0)


def frame_with_formants(formants_hz, base=None) -> torch.Tensor:
    """A (1, 1, 13) frame with f1-f4 set to the given frequencies."""
    frame = torch.rand(1, 1, 13) if base is None else base.clone()
    for name, hz in zip(FORMANT_NAMES, formants_hz):
        frame[..., PARAM_NAMES.index(name)] = hz / (BYTE_SCALE * FORMANT_SCALE)
    return frame


def poles(coeffs: torch.Tensor) -> np.ndarray:
    """Roots of 1 - sum a_i z^-i for one coefficient vector."""
    return np.roots(np.concatenate([[1.0], -coeffs.detach().numpy().ravel()]))


def resonances(coeffs: torch.Tensor):
    """(frequencies Hz, bandwidths Hz) of the complex pole pairs, sorted by frequency."""
    r = poles(coeffs)
    r = r[r.imag > 1e-9]
    order = np.argsort(np.angle(r))
    r = r[order]
    return np.angle(r) * SR / (2 * np.pi), -np.log(np.abs(r)) * SR / np.pi


class TestShape:
    @pytest.mark.parametrize("shape,expected", [((2, 50, 13), (2, 50, 10)),
                                                ((50, 13), (1, 50, 10)),
                                                ((13,), (1, 1, 10))])
    def test_shapes(self, shape, expected):
        assert FormantToLPC()(torch.rand(*shape)).shape == expected

    def test_float64(self):
        assert FormantToLPC()(torch.rand(1, 2, 13)).dtype == torch.float64

    def test_order_follows_tilt_poles(self):
        assert FormantToLPC(tilt_poles=0)(torch.rand(1, 1, 13)).shape[-1] == 8
        assert FormantToLPC(tilt_poles=3).order == 11

    def test_rejects_4d(self):
        with pytest.raises(ValueError):
            FormantToLPC()(torch.rand(1, 1, 1, 13))


class TestResonances:
    @pytest.mark.parametrize("vowel", [VOWEL_AH, VOWEL_EE])
    def test_poles_at_formant_frequencies(self, vowel):
        freqs, bandwidths = resonances(FormantToLPC()(frame_with_formants(vowel)))
        np.testing.assert_allclose(freqs, vowel, atol=0.5)
        np.testing.assert_allclose(bandwidths, 100.0, atol=0.5)

    def test_bandwidths_follow_initialization(self):
        mapper = FormantToLPC(bandwidths_hz=(60.0, 90.0, 150.0, 200.0))
        _, bandwidths = resonances(mapper(frame_with_formants(VOWEL_AH)))
        np.testing.assert_allclose(bandwidths, (60.0, 90.0, 150.0, 200.0), atol=0.5)
        np.testing.assert_allclose(mapper.bandwidths_hz().detach(), (60.0, 90.0, 150.0, 200.0))

    def test_magnitude_peaks_at_formants(self):
        a = FormantToLPC(tilt_poles=0)(frame_with_formants(VOWEL_AH)).detach().numpy().ravel()
        w, h = freqz([1.0], np.concatenate([[1.0], -a]), worN=8000, fs=SR)
        mag = np.abs(h)
        peaks = w[1:-1][(mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])]
        np.testing.assert_allclose(peaks, VOWEL_AH, atol=15)

    def test_moving_f2_moves_only_second_formant(self):
        """The bug the MLP mapper had: formants must move independently."""
        mapper = FormantToLPC()
        ah = resonances(mapper(frame_with_formants(VOWEL_AH)))[0]
        moved = (VOWEL_AH[0], 2000.0, VOWEL_AH[2], VOWEL_AH[3])
        after = resonances(mapper(frame_with_formants(moved)))[0]
        np.testing.assert_allclose(after, moved, atol=0.5)
        np.testing.assert_allclose(after[[0, 2, 3]], ah[[0, 2, 3]], atol=0.5)

    def test_ignores_non_formant_params(self):
        mapper = FormantToLPC()
        frame = frame_with_formants(VOWEL_AH)
        other = frame_with_formants(VOWEL_AH, base=torch.rand(1, 1, 13))
        assert torch.allclose(mapper(frame), mapper(other))

    def test_tilt_poles_are_real(self):
        mapper = FormantToLPC()
        with torch.no_grad():
            mapper.raw_tilt.copy_(torch.tensor([0.5, -0.3], dtype=torch.float64))
        r = poles(mapper(frame_with_formants(VOWEL_AH)))
        real = np.sort(r[np.abs(r.imag) < 1e-9].real)
        np.testing.assert_allclose(real, np.sort(mapper.tilt().detach().numpy()), atol=1e-6)


class TestStability:
    def test_random_params_stable(self):
        torch.manual_seed(0)
        a = FormantToLPC()(torch.rand(4, 100, 13))
        radius = max(np.abs(poles(row)).max() for row in a.reshape(-1, 10))
        assert radius < 1.0

    def test_extreme_learned_values_stay_inside_bounds(self):
        """Bandwidth at its floor and saturated tilt: radius bounded by the documented limits."""
        mapper = FormantToLPC()
        with torch.no_grad():
            mapper.log_bandwidth_excess.fill_(-100.0)  # bandwidth -> MIN_BANDWIDTH_HZ
            mapper.raw_tilt.fill_(100.0)  # tanh -> 1
        torch.manual_seed(1)
        bound = max(np.exp(-np.pi * MIN_BANDWIDTH_HZ / SR), MAX_TILT_POLE)
        for row in mapper(torch.rand(1, 50, 13))[0]:
            assert np.abs(poles(row)).max() <= bound + 1e-6

    @pytest.mark.parametrize("value", [0.0, 0.01, 1.0])
    def test_extreme_formant_values_realize_stable_filter(self, value):
        """
        All four formants at 0 Hz clusters eight poles near z = 1, where the
        direct-form polynomial is ill-conditioned (numpy.roots even reports a
        root at 1.0007; the exact roots of the rounded coefficients are <= 0.997).
        What matters is the realized float64 filter: its impulse response decays.
        """
        coeffs = FormantToLPC()(torch.full((1, 1, 13), value)).detach()
        impulse = torch.zeros(1, 32000, dtype=torch.float64)
        impulse[0, 0] = 1.0
        response = LPCFilter()(impulse, coeffs.expand(1, 32000, -1))
        assert torch.isfinite(response).all()
        assert response[0, -1000:].abs().max() < 1e-12 * response.abs().max()


class TestGradients:
    def test_only_formant_params_get_gradient(self):
        params = torch.rand(2, 10, 13, requires_grad=True)
        FormantToLPC()(params).pow(2).sum().backward()
        formant_cols = [PARAM_NAMES.index(n) for n in FORMANT_NAMES]
        other_cols = [i for i in range(13) if i not in formant_cols]
        assert params.grad[..., formant_cols].abs().sum(dim=(0, 1)).min() > 0
        assert params.grad[..., other_cols].abs().sum() == 0

    def test_learned_values_get_gradient(self):
        mapper = FormantToLPC()
        mapper(torch.rand(1, 5, 13)).pow(2).sum().backward()
        assert mapper.log_bandwidth_excess.grad.abs().min() > 0
        assert mapper.raw_tilt.grad.abs().sum() > 0

    def test_gradcheck(self):
        mapper = FormantToLPC()
        params = torch.rand(1, 2, 13, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(mapper, (params,))


class TestConstruction:
    def test_rejects_bad_bandwidths(self):
        with pytest.raises(ValueError):
            FormantToLPC(bandwidths_hz=(100.0, 100.0, 100.0))
        with pytest.raises(ValueError):
            FormantToLPC(bandwidths_hz=(MIN_BANDWIDTH_HZ, 100.0, 100.0, 100.0))

    def test_rejects_negative_tilt_poles(self):
        with pytest.raises(ValueError):
            FormantToLPC(tilt_poles=-1)

    def test_make_mapper(self):
        assert isinstance(make_mapper("formant"), FormantToLPC)
        assert isinstance(make_mapper("mlp"), ParamToLPC)
        with pytest.raises(ValueError):
            make_mapper("other")

    def test_trainable_parameter_count(self):
        """Four bandwidths and two tilt poles: nothing that could replace the formant params."""
        assert sum(p.numel() for p in FormantToLPC().parameters()) == 6
