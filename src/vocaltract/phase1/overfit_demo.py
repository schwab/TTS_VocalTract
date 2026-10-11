"""
Phase 1 "hello world": overfit one 1-second clip by optimizing parameter frames.

The free variable is the (1, 50, 13) parameter tensor (and, optionally, the
``ParamToLPC`` mapper). Each step synthesizes audio with

    params -> ParamToLPC -> frames_to_samples -> LPCFilter
    params -> make_excitation ---------------------^

and minimizes ``MultiScaleSTFTLoss + l1_weight * L1(waveform)`` with Adam.

Targets come from ``assets/phase1`` (see its README):

- ``--target-mode wav`` (default, Step 7 option b): load the target WAV.
- ``--target-mode self`` (option a): render the target with this same pipeline
  from the target's ``.npz`` parameters and a fixed-seed mapper, so exact
  recovery is possible.

Design choices, found by experiment on the bundled targets:

- ``params`` is optimized directly and clamped to [0, 1] after every step
  (projected gradient descent). Many target parameters are exactly 0 (e.g. no
  aspiration); a sigmoid parameterization can never reach 0, and the
  log-magnitude loss heavily penalizes faint noise against silence.
- The default initialization is the target's ``.npz`` parameters, as the assets
  README recommends. Starting from a neutral value leaves pitch stuck in a
  local minimum: spectral losses give poor gradients for sinusoid frequency.
- Noise uses the same fixed seed every step, so the waveform L1 term has no
  irreducible noise floor.
- The mapper is frozen by default. Training it at high learning rates can drive
  its reflection coefficients to +/-1, where the filter gain overflows.

Usage:

    python -m vocaltract.phase1.overfit_demo
    python -m vocaltract.phase1.overfit_demo --target vt_glide_ah_ee --train-mapper
    python -m vocaltract.phase1.overfit_demo --target-mode self --init mid
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

from vocaltract.filter import LPCFilter, frames_to_samples
from vocaltract.losses import MultiScaleSTFTLoss
from vocaltract.mapper import ParamToLPC
from vocaltract.params import PARAM_DIM, denormalize
from vocaltract.source import make_excitation

SAMPLE_RATE = 16000
HOP = 320
NOISE_SEED = 0
DEFAULT_ASSETS = Path(__file__).resolve().parents[3] / "assets" / "phase1"
INIT_CHOICES = ("npz", "mid", "random")


@dataclass
class DemoConfig:
    """Settings for one overfit run (mirrors the command-line flags)."""

    target: str = "vt_vowel_ah"
    target_mode: str = "wav"
    init: Optional[str] = None  # None -> "npz" for wav targets, "mid" for self targets
    steps: int = 500
    lr: float = 1e-3
    l1_weight: float = 0.1
    train_mapper: bool = False
    log_every: int = 50
    seed: int = 0
    assets_dir: Path = DEFAULT_ASSETS
    out: Path = Path("out/phase1_overfit.wav")

    def resolved_init(self) -> str:
        init = self.init or ("npz" if self.target_mode == "wav" else "mid")
        if self.target_mode == "self" and init == "npz":
            raise ValueError("--target-mode self renders the target from the npz params; "
                             "use --init mid or random so there is something to recover")
        return init


@dataclass
class DemoResult:
    """Outcome of a run."""

    history: List[float]
    params: torch.Tensor
    truth_params: torch.Tensor
    initial_audio: torch.Tensor
    final_audio: torch.Tensor
    target: torch.Tensor
    diverged_at: Optional[int] = None
    seconds: float = 0.0
    log: List[str] = field(default_factory=list)

    @property
    def initial_loss(self) -> float:
        return self.history[0]

    @property
    def final_loss(self) -> float:
        return self.history[-1]


def list_targets(assets_dir: Path = DEFAULT_ASSETS) -> List[str]:
    """Target names from ``manifest.json``, in difficulty order."""
    manifest = json.loads((assets_dir / "manifest.json").read_text())
    return [entry["name"] for entry in manifest]


def load_asset(name: str, assets_dir: Path = DEFAULT_ASSETS) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a Phase 1 target.

    Returns:
        (audio (1, 16000) float32, params (1, 50, 13) float32 in [0, 1]).
    """
    audio, sample_rate = sf.read(assets_dir / f"{name}.wav", dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{name}.wav is {sample_rate} Hz, expected {SAMPLE_RATE}")
    params = np.load(assets_dir / f"{name}.npz")["params"]
    return torch.from_numpy(audio)[None], torch.from_numpy(params).float()[None]


def synthesize(params: torch.Tensor, mapper: ParamToLPC) -> torch.Tensor:
    """params (B, T_frames, 13) -> audio (B, T_frames * HOP) with fixed-seed noise."""
    noise = torch.Generator().manual_seed(NOISE_SEED)
    excitation = make_excitation(params, sample_rate=SAMPLE_RATE, hop=HOP, generator=noise)
    return LPCFilter()(excitation, frames_to_samples(mapper(params), HOP))


def initial_params(init: str, truth: torch.Tensor) -> torch.Tensor:
    """Starting point for the free parameter tensor."""
    if init == "npz":
        return truth.clone()
    if init == "mid":
        return torch.full_like(truth, 0.5)
    if init == "random":
        return torch.rand_like(truth)
    raise ValueError(f"unknown init {init!r}, expected one of {INIT_CHOICES}")


def run(cfg: DemoConfig) -> DemoResult:
    """Optimize params (and optionally the mapper) toward the target clip."""
    torch.manual_seed(cfg.seed)
    init = cfg.resolved_init()
    mapper = ParamToLPC().requires_grad_(cfg.train_mapper)

    wav, truth = load_asset(cfg.target, cfg.assets_dir)
    if cfg.target_mode == "wav":
        target = wav
    elif cfg.target_mode == "self":
        with torch.no_grad():
            target = synthesize(truth, mapper)
    else:
        raise ValueError(f"unknown target mode {cfg.target_mode!r}")

    params = initial_params(init, truth).requires_grad_()
    trainable = [params] + (list(mapper.parameters()) if cfg.train_mapper else [])
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)
    loss_fn = MultiScaleSTFTLoss()

    def objective() -> Tuple[torch.Tensor, torch.Tensor]:
        pred = synthesize(params, mapper)
        return loss_fn(pred, target) + cfg.l1_weight * (pred - target).abs().mean(), pred

    history, log = [], []
    initial_audio = None
    diverged_at = None
    start = time.perf_counter()
    for step in range(cfg.steps + 1):
        loss, pred = objective()
        if not torch.isfinite(loss):
            diverged_at = step
            log.append(f"step {step:4d}  loss is non-finite, stopping")
            break
        history.append(loss.item())
        if initial_audio is None:
            initial_audio = pred.detach()
        final_audio = pred.detach()
        if step % cfg.log_every == 0 or step == cfg.steps:
            log.append(f"step {step:4d}  loss {loss.item():.4f}")
        if step == cfg.steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            params.clamp_(0.0, 1.0)

    return DemoResult(
        history=history,
        params=params.detach(),
        truth_params=truth,
        initial_audio=initial_audio,
        final_audio=final_audio,
        target=target.detach(),
        diverged_at=diverged_at,
        seconds=time.perf_counter() - start,
        log=log,
    )


def median_pitch_hz(params: torch.Tensor) -> float:
    return denormalize(params)["gp"].median().item()


def window_trend(history: Sequence[float], window: int = 50) -> Tuple[float, float]:
    """Mean loss over the first and last ``window`` steps."""
    window = max(1, min(window, len(history) // 2))
    return float(np.mean(history[:window])), float(np.mean(history[-window:]))


def save_wavs(result: DemoResult, out: Path) -> List[Path]:
    """Write the final prediction to ``out``, plus target and initial audio beside it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        (out, result.final_audio),
        (out.with_name(f"{out.stem}_target.wav"), result.target),
        (out.with_name(f"{out.stem}_initial.wav"), result.initial_audio),
    ]
    for path, audio in paths:
        # FLOAT subtype: the untrained filter can exceed +/-1, which PCM would clip.
        sf.write(path, audio[0].numpy(), SAMPLE_RATE, subtype="FLOAT")
    return [p for p, _ in paths]


def parse_args(argv: Optional[Sequence[str]] = None) -> DemoConfig:
    defaults = DemoConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", default=defaults.target,
                        help="asset name from assets/phase1/manifest.json")
    parser.add_argument("--target-mode", choices=("wav", "self"), default=defaults.target_mode,
                        help="wav: load the WAV; self: render the target with this pipeline")
    parser.add_argument("--init", choices=INIT_CHOICES, default=None,
                        help="parameter initialization (default: npz for wav, mid for self)")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--l1-weight", type=float, default=defaults.l1_weight)
    parser.add_argument("--train-mapper", action="store_true", help="also optimize ParamToLPC")
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--assets-dir", type=Path, default=defaults.assets_dir)
    parser.add_argument("--out", type=Path, default=defaults.out)
    return DemoConfig(**vars(parser.parse_args(argv)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the demo, save WAVs, print a summary, and return a process exit code."""
    cfg = parse_args(argv)
    try:
        init = cfg.resolved_init()
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print(f"Phase 1 overfit: target={cfg.target} mode={cfg.target_mode} init={init} "
          f"steps={cfg.steps} lr={cfg.lr:g} train_mapper={cfg.train_mapper}")
    result = run(cfg)
    for line in result.log:
        print(f"  {line}")

    for path in save_wavs(result, cfg.out):
        print(f"  wrote {path}")

    first, last = window_trend(result.history)
    print(f"  initial loss {result.initial_loss:.4f} -> final loss {result.final_loss:.4f} "
          f"({result.seconds:.1f}s)")
    print(f"  mean loss first/last 50 steps: {first:.4f} -> {last:.4f}")
    print(f"  median gp: {median_pitch_hz(result.params):.1f} Hz "
          f"(npz params: {median_pitch_hz(result.truth_params):.1f} Hz)")

    if result.diverged_at is not None:
        print(f"FAIL: loss became non-finite at step {result.diverged_at}", file=sys.stderr)
        return 1
    if not result.final_loss < result.initial_loss:
        print("FAIL: final loss is not below initial loss", file=sys.stderr)
        return 1
    print("PASS: loss decreased")
    return 0


if __name__ == "__main__":
    sys.exit(main())
