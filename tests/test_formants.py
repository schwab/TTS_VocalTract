"""Tests for formant measurement from audio."""

import numpy as np
import torch

from vocaltract.formants import formant_error_hz, measure_formants
from vocaltract.mapper import FORMANT_NAMES, FormantToLPC
from vocaltract.params import BYTE_SCALE, FORMANT_SCALE, PARAM_NAMES
from vocaltract.pipeline import synthesize

GP_120HZ = (100 + 48 * np.log2(120 / 110)) / BYTE_SCALE


def vowel_params(start_hz, end_hz=None, frames: int = 50) -> torch.Tensor:
    """Voiced frames (no noise) with formants gliding linearly from start to end."""
    end_hz = end_hz or start_hz
    params = torch.zeros(1, frames, 13)
    params[..., PARAM_NAMES.index("ga")] = 0.8
    params[..., PARAM_NAMES.index("gp")] = GP_120HZ
    ramp = torch.linspace(0, 1, frames)
    for name, a, b in zip(FORMANT_NAMES, start_hz, end_hz):
        params[0, :, PARAM_NAMES.index(name)] = (a + (b - a) * ramp) / (BYTE_SCALE * FORMANT_SCALE)
    return params


def render(params: torch.Tensor) -> torch.Tensor:
    """All-pole render: LPC analysis is exact for all-pole signals, so this tests the estimator."""
    mapper = FormantToLPC(tilt_poles=0, complex_resonators=False)
    with torch.no_grad():
        return synthesize(params, mapper, generator=torch.Generator().manual_seed(0))


def test_measures_steady_vowel():
    audio = render(vowel_params((730, 1090, 2440, 3400)))
    formants = measure_formants(audio)
    assert formants.shape == (49, 3)
    median = np.nanmedian(formants[5:], axis=0)
    np.testing.assert_allclose(median, (730, 1090, 2440), rtol=0.05)


def test_tracks_glide():
    """/a/ -> /i/: F1 falls, F2 rises; measured tracks follow."""
    audio = render(vowel_params((730, 1090, 2440, 3400), (300, 2290, 3010, 3500)))
    formants = measure_formants(audio)
    start, end = np.nanmedian(formants[3:8], axis=0), np.nanmedian(formants[-6:-1], axis=0)
    assert start[1] < 1300 and end[1] > 2100
    assert end[0] < start[0] - 250


def test_silence_is_nan():
    assert np.isnan(measure_formants(np.zeros(16000))).all()


def test_frame_count_and_hop():
    assert measure_formants(np.random.default_rng(0).standard_normal(16000), hop=160).shape[0] == 98


def test_error_zero_for_identical_and_large_for_different_vowels():
    ah = render(vowel_params((730, 1090, 2440, 3400)))
    ee = render(vowel_params((300, 2290, 3010, 3500)))
    np.testing.assert_allclose(formant_error_hz(ah, ah), 0.0)
    assert formant_error_hz(ah, ee)[1] > 800


def test_error_nan_without_estimates():
    assert np.isnan(formant_error_hz(np.zeros(16000), np.zeros(16000))).all()
