"""Unit tests for the multi-scale STFT loss."""

import pytest
import torch

from vocaltract.losses import (
    MultiScaleSTFTLoss,
    log_magnitude_l1,
    spectral_convergence,
    stft_magnitude,
)

SR = 16000


def tone(freq: float, n: int = SR, batch: int = 1) -> torch.Tensor:
    t = torch.arange(n) / SR
    return torch.sin(2 * torch.pi * freq * t).expand(batch, n).clone()


class TestOutput:
    def test_scalar_with_grad(self):
        torch.manual_seed(0)
        pred = torch.randn(2, SR, requires_grad=True)
        loss = MultiScaleSTFTLoss()(pred, torch.randn(2, SR))
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_no_grad_without_grad_inputs(self):
        loss = MultiScaleSTFTLoss()(torch.randn(1, 4096), torch.randn(1, 4096))
        assert not loss.requires_grad

    def test_accepts_1d(self):
        x, y = torch.randn(4096), torch.randn(4096)
        loss_fn = MultiScaleSTFTLoss()
        assert torch.allclose(loss_fn(x, y), loss_fn(x[None], y[None]))

    @pytest.mark.parametrize("pred_shape, target_shape", [((1, 4096), (1, 4000)), ((1, 1, 4096),) * 2])
    def test_bad_shapes_raise(self, pred_shape, target_shape):
        with pytest.raises(ValueError):
            MultiScaleSTFTLoss()(torch.randn(*pred_shape), torch.randn(*target_shape))


class TestConfig:
    def test_defaults(self):
        loss_fn = MultiScaleSTFTLoss()
        assert loss_fn.ffts == (512, 1024, 2048)
        assert loss_fn.hops == (128, 256, 512)
        assert loss_fn.win_lengths == (512, 1024, 2048)

    def test_custom_scales(self):
        loss_fn = MultiScaleSTFTLoss(ffts=(256, 512), hops=(64, 100), win_lengths=(200, 512))
        loss = loss_fn(torch.randn(1, 4096), torch.randn(1, 4096))
        assert torch.isfinite(loss)

    @pytest.mark.parametrize(
        "kwargs",
        [dict(ffts=(512, 1024), hops=(128,)), dict(ffts=(512,), win_lengths=(1024,))],
    )
    def test_bad_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            MultiScaleSTFTLoss(**kwargs)


class TestValues:
    def test_zero_for_identical(self):
        x = torch.randn(2, SR)
        assert MultiScaleSTFTLoss()(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_time_shift_cheaper_than_pitch_change(self):
        """A time shift costs far less than a pitch change (the log term still sees sidelobe noise)."""
        target = tone(440.0)
        shifted = torch.roll(target, 7, dims=-1)
        loss_fn = MultiScaleSTFTLoss()
        assert loss_fn(shifted, target) < 0.2 * loss_fn(tone(880.0), target)

    def test_closer_pitch_lower_loss(self):
        target = tone(440.0)
        loss_fn = MultiScaleSTFTLoss()
        assert loss_fn(tone(450.0), target) < loss_fn(tone(600.0), target)

    def test_finite_on_silence(self):
        """EPS clamping keeps log-magnitude and gradients finite on all-zero audio."""
        pred = torch.zeros(1, 4096, requires_grad=True)
        loss = MultiScaleSTFTLoss()(pred, torch.zeros(1, 4096))
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(pred.grad).all()

    def test_finite_grad_on_denormal_audio(self):
        """
        Near-silent predictions (float32 denormals, ~1e-40) make |z| round to 0
        while z != 0; the gradient z/|z| must not become inf/NaN.
        """
        torch.manual_seed(0)
        pred = (torch.randn(1, 4096) * 1e-40).requires_grad_()
        MultiScaleSTFTLoss()(pred, torch.randn(1, 4096)).backward()
        assert torch.isfinite(pred.grad).all()

    def test_components(self):
        mag = stft_magnitude(torch.randn(2, 4096), n_fft=512, hop=128, win_length=512)
        assert mag.shape == (2, 257, 4096 // 128 + 1)
        assert spectral_convergence(mag, mag).item() == 0.0
        assert log_magnitude_l1(mag, mag).item() == 0.0
        # SC is relative: scaling both spectra leaves it unchanged
        other = mag * 1.5
        assert torch.allclose(spectral_convergence(other, mag), spectral_convergence(3 * other, 3 * mag))


class TestGradients:
    def test_grads_reach_pred_and_target(self):
        """Neither input is detached."""
        torch.manual_seed(0)
        pred = torch.randn(1, 4096, requires_grad=True)
        target = torch.randn(1, 4096, requires_grad=True)
        MultiScaleSTFTLoss()(pred, target).backward()
        for t in (pred, target):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0
