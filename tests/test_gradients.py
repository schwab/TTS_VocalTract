"""Gradient-flow and single-optimizer-step tests for the Phase 1 pipeline."""

import torch

from vocaltract.filter import LPCFilter, frames_to_samples
from vocaltract.losses import MultiScaleSTFTLoss
from vocaltract.mapper import ParamToLPC
from vocaltract.source import make_excitation

HOP = 160


def synthesize(params: torch.Tensor, mapper: ParamToLPC, seed: int = 0) -> torch.Tensor:
    """params (B, T_frames, 13) -> audio (B, T_frames * HOP), with fixed-seed noise."""
    excitation = make_excitation(params, hop=HOP, generator=torch.Generator().manual_seed(seed))
    coeffs = frames_to_samples(mapper(params), HOP)
    return LPCFilter()(excitation, coeffs)


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
    def test_grads_reach_params(self):
        torch.manual_seed(0)
        params = torch.rand(1, 20, 13, requires_grad=True)
        target = torch.randn(1, 20 * HOP)
        MultiScaleSTFTLoss()(synthesize(params, ParamToLPC()), target).backward()
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()
        assert params.grad.abs().sum() > 0

    def test_one_step_reduces_loss(self):
        """One Adam step on params moves the synthesized audio toward a target clip."""
        torch.manual_seed(0)
        mapper = ParamToLPC()
        target = synthesize(torch.rand(1, 20, 13), mapper, seed=1).detach()

        params = torch.rand(1, 20, 13, requires_grad=True)
        loss_fn = MultiScaleSTFTLoss()
        optimizer = torch.optim.Adam([params], lr=1e-3)

        initial = loss_fn(synthesize(params, mapper), target)
        optimizer.zero_grad()
        initial.backward()
        optimizer.step()
        after = loss_fn(synthesize(params, mapper), target)
        assert after.item() < initial.item()
