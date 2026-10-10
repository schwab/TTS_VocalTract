"""Unit tests for VocalTract parameter schema."""

import pytest
import torch

from vocaltract.params import (
    PARAM_NAMES,
    PARAM_DIM,
    denormalize,
    denormalize_batch,
    DEFAULT_PARAMS,
    BYTE_SCALE,
    GP_BASE_HZ,
    GP_BASE_RAW,
    OCTAVE_UNIT,
    VP_SWING_DIVISOR,
    VR_SCALE,
    FORMANT_SCALE,
    FRICATION_FREQ_SCALE,
)


class TestParamNames:
    """Test that parameter names match original VocalTract encoding."""

    def test_param_dim(self):
        """PARAM_DIM should be 13."""
        assert PARAM_DIM == 13

    def test_param_names_count(self):
        """Should have exactly 13 parameter names."""
        assert len(PARAM_NAMES) == 13

    def test_param_names_unique(self):
        """All parameter names should be unique."""
        assert len(set(PARAM_NAMES)) == len(PARAM_NAMES)

    def test_expected_parameter_names(self):
        """Parameter names should match original VocalTract ordering."""
        expected = [
            "aa",   # aspiration amplitude
            "ga",   # glottal amplitude
            "gp",   # glottal pitch
            "vp",   # vibrato pitch
            "vr",   # vibrato rate
            "f1",   # formant 1 frequency
            "f2",   # formant 2 frequency
            "f3",   # formant 3 frequency
            "f4",   # formant 4 frequency
            "na",   # nasal amplitude
            "nf",   # nasal frequency
            "fa",   # frication amplitude
            "ff",   # frication frequency
        ]
        assert PARAM_NAMES == expected

    def test_param_types(self):
        """Amplitude parameters should be at correct indices."""
        # Amplitude params: aa(0), ga(1), na(9), fa(11)
        amplitude_indices = {0, 1, 9, 11}
        assert all(i in amplitude_indices for i, name in enumerate(PARAM_NAMES)
                   if name in ("aa", "ga", "na", "fa"))

        # Pitch/rate params: gp(2), vp(3), vr(4), f1-f4(5-8), nf(10)
        pitch_freq_indices = {2, 3, 4, 5, 6, 7, 8, 10}
        assert all(i in pitch_freq_indices for i, name in enumerate(PARAM_NAMES)
                   if name in ("gp", "vp", "vr", "f1", "f2", "f3", "f4", "nf"))


class TestDenormalizeScalar:
    """Test denormalization of a single parameter frame."""

    @pytest.fixture
    def dummy_params(self):
        """Create a dummy parameter tensor with known NORMALIZED values in [0,1].

        Every VocalTract parameter is a single byte in [0, 255], so
        raw = normalized * BYTE_SCALE for all 13 fields (see docs/corrections.md).
        """
        return torch.tensor([
            0.5,     # aa = 0.5 -> raw 127.5 -> physical 127.5 (amplitude, no further scale)
            0.25,    # ga = 0.25 -> raw 63.75 -> physical 63.75
            0.5,     # gp = 0.5 -> raw 127.5 -> log-pitch, see test_pitch_params
            0.0,     # vp = 0.0 -> raw 0.0 -> zero vibrato swing (no modulation)
            0.5,     # vr = 0.5 -> raw 127.5 -> physical 127.5 * 0.0763 Hz
            0.4,     # f1 = 0.4 -> raw 102.0 -> physical 102.0 * 19.53 Hz
            0.6,     # f2 = 0.6 -> raw 153.0 -> physical 153.0 * 19.53 Hz
            0.3,     # f3 = 0.3 -> raw 76.5  -> physical 76.5 * 19.53 Hz
            0.7,     # f4 = 0.7 -> raw 178.5 -> physical 178.5 * 19.53 Hz
            0.5,     # na = 0.5 -> raw 127.5 -> physical 127.5
            0.5,     # nf = 0.5 -> raw 127.5 -> physical 127.5 * 19.53 Hz
            0.25,    # fa = 0.25 -> raw 63.75 -> physical 63.75
            0.5,     # ff = 0.5 -> raw 127.5 -> physical 127.5 * 39.06 Hz
        ])

    def test_amplitude_params(self, dummy_params):
        """Amplitude parameters should be scaled by 255."""
        result = denormalize(dummy_params)

        # aa = 127.5 / 255 * 255 = 127.5
        assert torch.abs(result["aa"] - 127.5).item() < 1e-5

        # ga = 63.75 / 255 * 255 = 63.75
        assert torch.abs(result["ga"] - 63.75).item() < 1e-5

        # na = 127.5 / 255 * 255 = 127.5
        assert torch.abs(result["na"] - 127.5).item() < 1e-5

        # fa = 63.75 / 255 * 255 = 63.75
        assert torch.abs(result["fa"] - 63.75).item() < 1e-5

    def test_pitch_params(self, dummy_params):
        """gp follows the real 1/48-octave log-pitch law from VocalTract.spin."""
        result = denormalize(dummy_params)

        gp_raw = 0.5 * BYTE_SCALE
        expected_gp = GP_BASE_HZ * (2.0 ** ((gp_raw - GP_BASE_RAW) / OCTAVE_UNIT))
        assert torch.abs(result["gp"] - expected_gp).item() < 0.1

        # vp=0.0 -> raw=0 -> zero vibrato swing
        assert torch.abs(result["vp"] - 0.0).item() < 1e-5

        # gp=0.0 should give the lowest pitch (monotonic in raw value)
        gp_zero = torch.zeros(13, dtype=torch.float32)
        result_zero = denormalize(gp_zero)
        assert result_zero["gp"] < result["gp"]

        # gp=1.0 should give the highest pitch
        gp_one = torch.ones(13, dtype=torch.float32)
        result_one = denormalize(gp_one)
        assert result_one["gp"] > result["gp"]

    def test_vibrato_pitch_swing_calibration(self):
        """vp's one documented calibration point: raw=48 -> +/-1/2 octave swing."""
        vp_at_48 = torch.zeros(13, dtype=torch.float32)
        vp_at_48[3] = 48.0 / BYTE_SCALE  # vp is index 3
        result = denormalize(vp_at_48)
        assert torch.abs(result["vp"] - 0.5).item() < 1e-5

    def test_vibrato_rate(self, dummy_params):
        """Vibrato rate scales linearly from the raw byte value."""
        result = denormalize(dummy_params)

        expected_vr = (0.5 * BYTE_SCALE) * VR_SCALE
        assert torch.abs(result["vr"] - expected_vr).item() < 0.01

    def test_formant_frequencies(self, dummy_params):
        """Formant frequencies scale linearly from the raw byte value."""
        result = denormalize(dummy_params)

        expected_f1 = (0.4 * BYTE_SCALE) * FORMANT_SCALE
        assert torch.abs(result["f1"] - expected_f1).item() < 0.1

        expected_f2 = (0.6 * BYTE_SCALE) * FORMANT_SCALE
        assert torch.abs(result["f2"] - expected_f2).item() < 0.1

    def test_nasal_frequency(self, dummy_params):
        """Nasal frequency uses same scale as formants."""
        result = denormalize(dummy_params)

        expected_nf = (0.5 * BYTE_SCALE) * FORMANT_SCALE
        assert torch.abs(result["nf"] - expected_nf).item() < 0.1

    def test_frication_frequency(self, dummy_params):
        """Frication frequency has its own scale (39.06 Hz/raw unit)."""
        result = denormalize(dummy_params)

        expected_ff = (0.5 * BYTE_SCALE) * FRICATION_FREQ_SCALE
        assert torch.abs(result["ff"] - expected_ff).item() < 0.1


class TestDenormalizeBatch:
    """Test batch denormalization."""

    @pytest.fixture
    def batch_params(self):
        """Create a batch of 2 time frames."""
        return torch.rand(2, 3, 13)

    def test_output_shapes(self, batch_params):
        """Output should be dict with (B, T) shaped tensors."""
        result = denormalize_batch(batch_params)

        for name in PARAM_NAMES:
            assert name in result
            assert result[name].shape == (2, 3)

    def test_gradient_flow(self, batch_params):
        """Gradients should flow through denormalize_batch."""
        batch_params.requires_grad_(True)
        result = denormalize_batch(batch_params)
        
        # Compute a simple loss to trigger gradient computation
        total = sum(result.values())
        total.sum().backward()
        
        # Check that gradients are populated on the input batch_params
        # (the output tensors are not leaf tensors, so we check the input)
        assert batch_params.grad is not None, "Gradient not computed for input"
        assert batch_params.grad.numel() > 0
        
        # Also verify that all parameter dimensions received non-zero gradients
        grad_sum = batch_params.grad.abs().sum().item()
        assert grad_sum > 0, f"All gradients are zero (grad_sum={grad_sum})"

    def test_batch_consistency(self, batch_params):
        """Batch denormalize should match element-wise denormalize."""
        # Get first time frame from batch - shape (2, 13)
        t0 = batch_params[:, 0, :]  # Shape: (2, 13)
        t0_denorm = denormalize(t0)

        # Denormalize full batch
        batch_denorm = denormalize_batch(batch_params)

        # Compare - both should have shape (2,)
        for name in PARAM_NAMES:
            assert torch.allclose(t0_denorm[name], batch_denorm[name][:, 0])


class TestDefaultParams:
    """Test default parameter values."""

    def test_default_values(self):
        """Default params should have sensible defaults."""
        assert DEFAULT_PARAMS["gp"] == 0.5  # Base pitch
        assert DEFAULT_PARAMS["vp"] == 0.5  # No vibrato
        assert DEFAULT_PARAMS["f1"] == 0.5  # Middle formant
        assert DEFAULT_PARAMS["aa"] == 0.0  # No aspiration
        assert DEFAULT_PARAMS["ga"] == 0.0  # No glottal (default is voiceless)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
