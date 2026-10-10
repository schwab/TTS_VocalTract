"""
Gradient-flow smoke test for the full Phase 1 pipeline.

Runs one forward/backward pass through every stage:

    params (1, 50, 13) --ParamToLPC--> coeffs (1, 50, order)
                       --frames_to_samples--> coeffs (1, 50*hop, order)
    params --make_excitation--> excitation (1, 50*hop)
    (excitation, coeffs) --LPCFilter--> audio (1, 50*hop)
    loss = audio.pow(2).mean();  loss.backward()

and checks that a non-zero, finite gradient reaches ``params``. Prints a shape
trace, the loss and the gradient norm, and exits non-zero on failure so it can
be used in CI:

    python -m vocaltract.phase1.smoke_test
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from vocaltract.filter import LPCFilter, frames_to_samples
from vocaltract.mapper import ParamToLPC
from vocaltract.params import PARAM_DIM
from vocaltract.source import make_excitation

NUM_FRAMES = 50
HOP = 320
SAMPLE_RATE = 16000


@dataclass
class SmokeResult:
    """Outcome of one smoke-test pass."""

    shapes: Dict[str, Tuple[int, ...]]
    loss: float
    grad_norm: float
    failures: List[str]

    @property
    def ok(self) -> bool:
        return not self.failures


def run_smoke_test(seed: int = 0, num_frames: int = NUM_FRAMES, hop: int = HOP) -> SmokeResult:
    """
    Run params -> mapper -> source -> filter -> audio -> loss -> backward.

    Args:
        seed: Seed for ``torch.manual_seed`` (params, mapper init, noise).
        num_frames: Number of parameter frames.
        hop: Samples per frame.

    Returns:
        A :class:`SmokeResult` with shapes, loss, gradient norm and any failures.
    """
    torch.manual_seed(seed)
    params = torch.rand(1, num_frames, PARAM_DIM, requires_grad=True)
    mapper = ParamToLPC()

    frame_coeffs = mapper(params)
    sample_coeffs = frames_to_samples(frame_coeffs, hop)
    excitation = make_excitation(params, sample_rate=SAMPLE_RATE, hop=hop)
    audio = LPCFilter()(excitation, sample_coeffs)
    loss = audio.pow(2).mean()
    loss.backward()

    shapes = {
        "params": tuple(params.shape),
        "lpc coeffs (frame)": tuple(frame_coeffs.shape),
        "lpc coeffs (sample)": tuple(sample_coeffs.shape),
        "excitation": tuple(excitation.shape),
        "audio": tuple(audio.shape),
    }
    grad = params.grad
    return SmokeResult(
        shapes=shapes,
        loss=loss.item(),
        grad_norm=grad.norm().item() if grad is not None else 0.0,
        failures=_check(audio, loss, grad, mapper),
    )


def _check(
    audio: torch.Tensor, loss: torch.Tensor, grad: torch.Tensor, mapper: ParamToLPC
) -> List[str]:
    """Collect failure messages (explicit checks, so they survive ``python -O``)."""
    failures = []
    if not torch.isfinite(audio).all():
        failures.append("audio contains NaN/Inf")
    if not torch.isfinite(loss):
        failures.append("loss is not finite")
    if grad is None:
        failures.append("params.grad is None")
    elif not torch.isfinite(grad).all():
        failures.append("params.grad contains NaN/Inf")
    elif grad.abs().sum() <= 0:
        failures.append("params.grad is all zero")
    if all(p.grad is None or p.grad.abs().sum() == 0 for p in mapper.parameters()):
        failures.append("mapper received no gradient")
    return failures


def main() -> int:
    """Run the smoke test, print a summary, and return a process exit code."""
    result = run_smoke_test()

    print("Phase 1 smoke test: params -> mapper -> source -> filter -> audio")
    for name, shape in result.shapes.items():
        print(f"  {name:<20} {shape}")
    print(f"  loss                 {result.loss:.6g}")
    print(f"  ||params.grad||      {result.grad_norm:.6g}")

    if not result.ok:
        for failure in result.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: gradients flow end to end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
