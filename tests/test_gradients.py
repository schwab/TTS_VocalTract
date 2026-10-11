"""Gradient-flow and single-optimizer-step tests for the Phase 1 pipeline."""

import pytest
import torch

from vocaltract.losses import MultiScaleSTFTLoss
from vocaltract.mapper import make_mapper
from vocaltract.params import PARAM_NAMES
from vocaltract.pipeline import synthesize as pipeline_synthesize

HOP = 160
GP = PARAM_NAMES.index("gp")


def synthesize(params: torch.Tensor, mapper: torch.nn.Module, seed: int = 0) -> torch.Tensor:
    """params (B, T_frames, 13) -> audio (B, T_frames * HOP), with fixed-seed noise."""
    return pipeline_synthesize(params, mapper, hop=HOP, generator=torch.Generator().manual_seed(seed))


class TestToyProblem:
    def test_one_step_reduces_loss_gain(self):
        """Learning a gain on a fixed signal: one Adam step lowers the STFT loss."""
        torch.manual_seed(0)
        target = torch.randn(1, 8000)
        log_gain = torch.zeros(1, requires_grad=True)
        loss_fn = MultiScaleSTFTLoss()
        optimizer = torch.optim.Adam([log_gain], lr=1e-1)

        def loss() -> torch.Tensor:
            return loss_fn(0.2 * log_gain.exp() * target, target)

        initial = loss()
        optimizer.zero_grad()
        initial.backward()
        optimizer.step()
        assert loss().item() < initial.item()


class TestEndToEnd:
    @pytest.mark.parametrize("kind", ["formant", "mlp"])
    def test_grads_reach_params(self, kind):
        torch.manual_seed(0)
        params = torch.rand(1, 20, 13, requires_grad=True)
        target = torch.randn(1, 20 * HOP)
        MultiScaleSTFTLoss()(synthesize(params, make_mapper(kind)), target).backward()
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()
        assert params.grad.abs().sum() > 0

    @pytest.mark.parametrize("kind", ["formant", "mlp"])
    @pytest.mark.parametrize("seed", range(5))
    def test_one_step_reduces_loss(self, seed, kind):
        """
        One Adam step on params moves the synthesized audio toward a target clip.

        Pitch is held fixed, as it is in practice given its own much smaller
        step: phase is a running sum, so a pitch change in an early frame
        time-shifts everything after it, and the loss is only locally linear in
        gp over ~1e-7. The overfit demo uses --pitch-lr-scale for this reason.
        """
        torch.manual_seed(seed)
        mapper = make_mapper(kind)
        target = synthesize(torch.rand(1, 20, 13), mapper, seed=1).detach()

        params = torch.rand(1, 20, 13, requires_grad=True)
        loss_fn = MultiScaleSTFTLoss()
        optimizer = torch.optim.Adam([params], lr=1e-3)

        initial = loss_fn(synthesize(params, mapper), target)
        optimizer.zero_grad()
        initial.backward()
        params.grad[..., GP] = 0.0
        optimizer.step()
        after = loss_fn(synthesize(params, mapper), target)
        assert after.item() < initial.item()
