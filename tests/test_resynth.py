"""Tests for level A analysis/resynthesis (vocaltract.analysis)."""

import numpy as np
import pytest
from scipy import signal

pytest.importorskip("parselmouth")

from vocaltract.analysis.resynth import (HOP_S, _fill_gaps, _overflowed, export_spin,
                                         gp_to_pitch, pitch_to_gp, resynthesize)
from vocaltract.reference.engine import AA, F1, FA, GA, GP

SR = 16000
F0 = 150.0
FORMANTS = (700.0, 1200.0, 2600.0, 3500.0)


def synthetic_vowel(seconds=1.0, silence=0.4):
    """Classic source-filter vowel: pulse train through 4 resonators, then silence."""
    n = int(seconds * SR)
    src = np.zeros(n)
    src[::int(SR / F0)] = 1.0
    y = src
    for f, bw in zip(FORMANTS, (80, 100, 150, 200)):
        r = np.exp(-np.pi * bw / SR)
        y = signal.lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r], y)
    y = y / np.abs(y).max() * 0.5
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1e-4, int(silence * SR))           # room noise
    return np.concatenate([quiet, y, quiet])


def test_pitch_mapping_round_trip():
    assert pitch_to_gp(np.array([110.0]))[0] == pytest.approx(100.0)
    assert pitch_to_gp(np.array([220.0]))[0] == pytest.approx(148.0)
    f = np.array([80.0, 150.0, 400.0])
    np.testing.assert_allclose(gp_to_pitch(pitch_to_gp(f)), f)


def test_fill_gaps_prefers_next_voiced_value():
    x = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    np.testing.assert_array_equal(_fill_gaps(x, prefer_next=True), [1, 1, 4, 4, 4, 4])
    np.testing.assert_allclose(_fill_gaps(x), [1, 1, 2, 3, 4, 4])


@pytest.fixture(scope="module")
def vowel_resynth():
    audio = synthetic_vowel()
    return resynthesize(audio, SR, "vowel"), audio


def test_recovers_pitch_and_formants(vowel_resynth):
    res, _ = vowel_resynth
    t = (np.arange(len(res.frames)) + 1) * HOP_S
    steady = (t > 0.6) & (t < 1.2)
    assert np.median(res.frames[steady, GP]) == pytest.approx(pitch_to_gp(np.array([F0]))[0], abs=1)
    for i in range(2):                                         # F1, F2: 19.53 Hz bytes
        got = np.median(res.frames[steady, F1 + i]) * 19.53
        assert got == pytest.approx(FORMANTS[i], abs=80)
    assert (res.frames[steady, GA] > 0).all()


def test_silence_stays_silent(vowel_resynth):
    res, _ = vowel_resynth
    t = (np.arange(len(res.frames)) + 1) * HOP_S
    quiet = (t < 0.35) | (t > 1.5)
    assert not res.frames[quiet][:, [GA, AA, FA]].any()


def test_no_overflow_and_level_follows_recording(vowel_resynth):
    """Close F1/F2 at a steady loud level overflows VocalTract's 32-bit math, so
    the gain backs off (retries are expected here) until the render is clean."""
    res, _ = vowel_resynth
    assert not _overflowed(res.frames.astype(np.int64), res.audio)
    out = res.audio.astype(np.float64)
    n = int(0.3 * 20000)
    assert np.abs(out[:n]).max() == 0                          # leading silence
    assert np.sqrt(np.mean(out[int(0.7 * 20000):int(1.1 * 20000)] ** 2)) > 1e6


def test_spin_export(tmp_path, vowel_resynth):
    res, _ = vowel_resynth
    path = export_spin(res.frames, tmp_path / "p.spin", "vowel")
    text = path.read_text()
    assert f"frame_count = {len(res.frames)}" in text
    assert text.count("        byte ") == len(res.frames)
    assert "v.go(14)" in text
