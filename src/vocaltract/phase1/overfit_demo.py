"""
Phase 1 "hello world": overfit one 1-second clip by optimizing parameter frames.

The free variable is the (1, 50, 13) parameter tensor and, by default for WAV
targets, the ``ParamToLPC`` mapper. Each step synthesizes audio with
:func:`vocaltract.pipeline.synthesize` (harmonic source, gain-normalized LPC
filter) and minimizes ``MultiScaleSTFTLoss + l1_weight * L1(waveform)`` with
Adam.

Targets come from ``assets/phase1`` (see its README):

- ``--target-mode wav`` (default, Step 7 option b): load the target WAV.
- ``--target-mode self`` (option a): render the target with this same pipeline
  from the target's ``.npz`` parameters and a frozen fixed-seed mapper, start
  from perturbed parameters, and report how far optimization moves them back
  toward the truth.

Design choices, found by experiment on the bundled targets:

- ``params`` is optimized directly and clamped to [0, 1] after every step
  (projected gradient descent). Many target parameters are exactly 0; a
  sigmoid parameterization can never reach 0, and the log-magnitude loss
  heavily penalizes faint noise against silence.
- Pitch (``gp``) gets its own, smaller step size (``--pitch-lr-scale``). Adam
  moves every parameter by about ``lr`` per step; at lr 1e-3 that is ~6 cents
  of pitch, which shifts high harmonics by more than an STFT bin and makes
  pitch random-walk away even from the correct value.
- The learning rate follows a cosine decay and the best step is kept: the
  loss landscape is sharp enough that a constant-rate Adam never settles.
- The mapper is trained by default on WAV targets. Its random initialization
  bears no relation to the f1-f4 formant parameters, so a frozen mapper
  cannot place the target's formants (log-spectral distance stays ~30 dB).
- Default initialization is the target's ``.npz`` parameters (assets README);
  from a neutral start pitch gets stuck in a local minimum of the spectral loss.
- Noise uses the same fixed seed every step, so the L1 term has no noise floor.
- ``--freeze aa=0,fa`` holds parameters fixed (optionally at a value), e.g. to
  stop the optimizer from faking harmonic energy with noise on vowel targets.

Usage:

    python -m vocaltract.phase1.overfit_demo
    python -m vocaltract.phase1.overfit_demo --target vt_glide_ah_ee --freeze aa=0
    python -m vocaltract.phase1.overfit_demo --target-mode self --perturb 0.01
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

from vocaltract.losses import MultiScaleSTFTLoss
from vocaltract.mapper import ParamToLPC
from vocaltract.params import PARAM_NAMES, denormalize
from vocaltract.pipeline import synthesize as pipeline_synthesize

SAMPLE_RATE = 16000
HOP = 320
NOISE_SEED = 0
DEFAULT_ASSETS = Path(__file__).resolve().parents[3] / "assets" / "phase1"
INIT_CHOICES = ("npz", "perturbed", "mid", "random")
GP_INDEX = PARAM_NAMES.index("gp")


@dataclass
class DemoConfig:
    """Settings for one overfit run (mirrors the command-line flags)."""

    target: str = "vt_vowel_ah"
    target_mode: str = "wav"
    init: Optional[str] = None  # None -> "npz" for wav targets, "perturbed" for self targets
    perturb: float = 0.05
    steps: int = 500
    lr: float = 1e-3
    pitch_lr_scale: float = 0.1
    lr_decay: bool = True
    l1_weight: float = 0.1
    train_mapper: Optional[bool] = None  # None -> True for wav targets, False for self targets
    freeze: Dict[str, Optional[float]] = field(default_factory=dict)
    log_every: int = 50
    seed: int = 0
    assets_dir: Path = DEFAULT_ASSETS
    out: Path = Path("out/phase1_overfit.wav")

    def resolved_init(self) -> str:
        init = self.init or ("npz" if self.target_mode == "wav" else "perturbed")
        if self.target_mode == "self" and init == "npz":
            raise ValueError("--target-mode self renders the target from the npz params; "
                             "use --init perturbed, mid or random so there is something to recover")
        return init

    def resolved_train_mapper(self) -> bool:
        if self.train_mapper is not None:
            return self.train_mapper
        return self.target_mode == "wav"


@dataclass
class DemoResult:
    """Outcome of a run. ``params``/``final_audio`` are from the best step."""

    history: List[float]
    best_step: int
    params: torch.Tensor
    initial_params: torch.Tensor
    truth_params: torch.Tensor
    initial_audio: torch.Tensor
    final_audio: torch.Tensor
    target: torch.Tensor
    frozen: List[str]
    diverged_at: Optional[int] = None
    seconds: float = 0.0
    log: List[str] = field(default_factory=list)

    @property
    def initial_loss(self) -> float:
        return self.history[0]

    @property
    def final_loss(self) -> float:
        """Best loss reached (the returned params and audio are from that step)."""
        return self.history[self.best_step]


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
    return pipeline_synthesize(params, mapper, hop=HOP, sample_rate=SAMPLE_RATE, generator=noise)


def parse_freeze(spec: str) -> Dict[str, Optional[float]]:
    """Parse ``"aa=0,fa"`` into ``{"aa": 0.0, "fa": None}`` (None: keep initial value)."""
    frozen: Dict[str, Optional[float]] = {}
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, _, value = item.partition("=")
        if name not in PARAM_NAMES:
            raise ValueError(f"unknown parameter {name!r} in --freeze; expected one of {PARAM_NAMES}")
        frozen[name] = float(value) if value else None
    return frozen


def initial_params(
    init: str, truth: torch.Tensor, perturb: float = 0.05, seed: int = 0
) -> torch.Tensor:
    """Starting point for the free parameter tensor."""
    if init == "npz":
        return truth.clone()
    if init == "perturbed":
        generator = torch.Generator().manual_seed(seed + 1)
        noise = 2 * torch.rand(truth.shape, generator=generator) - 1
        return (truth + perturb * noise).clamp(0.0, 1.0)
    if init == "mid":
        return torch.full_like(truth, 0.5)
    if init == "random":
        return torch.rand(truth.shape, generator=torch.Generator().manual_seed(seed + 1))
    raise ValueError(f"unknown init {init!r}, expected one of {INIT_CHOICES}")


class FrameParams:
    """
    The (1, T, 13) parameter frames, split so pitch gets its own Adam step size.

    ``pitch`` holds column ``gp``; ``rest`` holds the other 12 columns.
    """

    def __init__(self, start: torch.Tensor):
        self.pitch = start[..., GP_INDEX : GP_INDEX + 1].clone().requires_grad_()
        self.rest = torch.cat([start[..., :GP_INDEX], start[..., GP_INDEX + 1 :]], -1)
        self.rest = self.rest.clone().requires_grad_()

    def assemble(self) -> torch.Tensor:
        return torch.cat([self.rest[..., :GP_INDEX], self.pitch, self.rest[..., GP_INDEX:]], -1)

    def zero_grad_columns(self, names: Sequence[str]) -> None:
        for name in names:
            if name == "gp":
                self.pitch.grad.zero_()
            else:
                index = PARAM_NAMES.index(name)
                self.rest.grad[..., index - (index > GP_INDEX)] = 0.0

    @torch.no_grad()
    def clamp_(self) -> None:
        self.pitch.clamp_(0.0, 1.0)
        self.rest.clamp_(0.0, 1.0)


def run(cfg: DemoConfig) -> DemoResult:
    """Optimize params (and optionally the mapper) toward the target clip."""
    torch.manual_seed(cfg.seed)
    init = cfg.resolved_init()
    train_mapper = cfg.resolved_train_mapper()
    mapper = ParamToLPC().requires_grad_(train_mapper)

    wav, truth = load_asset(cfg.target, cfg.assets_dir)
    if cfg.target_mode == "wav":
        target = wav
    elif cfg.target_mode == "self":
        with torch.no_grad():
            target = synthesize(truth, mapper)
    else:
        raise ValueError(f"unknown target mode {cfg.target_mode!r}")

    start = initial_params(init, truth, cfg.perturb, cfg.seed)
    for name, value in cfg.freeze.items():
        if value is not None:
            start[..., PARAM_NAMES.index(name)] = value
    frames = FrameParams(start)

    groups = [
        {"params": [frames.rest], "lr": cfg.lr},
        {"params": [frames.pitch], "lr": cfg.lr * cfg.pitch_lr_scale},
    ]
    if train_mapper:
        groups.append({"params": list(mapper.parameters()), "lr": cfg.lr})
    optimizer = torch.optim.Adam(groups)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(cfg.steps, 1))
        if cfg.lr_decay else None
    )
    loss_fn = MultiScaleSTFTLoss()
    frozen = list(cfg.freeze)

    history, log = [], []
    best_step, best_params, best_audio, initial_audio = 0, start, None, None
    diverged_at = None
    begin = time.perf_counter()
    for step in range(cfg.steps + 1):
        params = frames.assemble()
        pred = synthesize(params, mapper)
        loss = loss_fn(pred, target) + cfg.l1_weight * (pred - target).abs().mean()
        if not torch.isfinite(loss):
            diverged_at = step
            log.append(f"step {step:4d}  loss is non-finite, stopping")
            break
        history.append(loss.item())
        if initial_audio is None:
            initial_audio = pred.detach()
        if history[-1] <= history[best_step] or best_audio is None:
            best_step, best_params, best_audio = step, params.detach().clone(), pred.detach()
        if step % cfg.log_every == 0 or step == cfg.steps:
            log.append(f"step {step:4d}  loss {loss.item():.4f}")
        if step == cfg.steps:
            break
        optimizer.zero_grad()
        loss.backward()
        frames.zero_grad_columns(frozen)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        frames.clamp_()

    return DemoResult(
        history=history,
        best_step=best_step,
        params=best_params,
        initial_params=start,
        truth_params=truth,
        initial_audio=initial_audio,
        final_audio=best_audio,
        target=target.detach(),
        frozen=frozen,
        diverged_at=diverged_at,
        seconds=time.perf_counter() - begin,
        log=log,
    )


def median_pitch_hz(params: torch.Tensor) -> float:
    return denormalize(params)["gp"].median().item()


def pitch_error_cents(params: torch.Tensor, truth: torch.Tensor) -> float:
    """Mean absolute per-frame pitch error in cents."""
    ratio = denormalize(params)["gp"] / denormalize(truth)["gp"]
    return (1200 * torch.log2(ratio)).abs().mean().item()


def param_error(params: torch.Tensor, truth: torch.Tensor, skip: Sequence[str] = ()) -> float:
    """Mean absolute normalized error over parameters not in ``skip``."""
    keep = [i for i, name in enumerate(PARAM_NAMES) if name not in skip]
    return (params[..., keep] - truth[..., keep]).abs().mean().item()


def log_spectral_distance(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024) -> float:
    """Mean per-frame RMS difference of log-magnitude spectra, in dB."""
    window = torch.hann_window(n_fft)

    def db(x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(x[0], n_fft, n_fft // 4, window=window, return_complex=True).abs()
        return 20 * spec.clamp_min(1e-5).log10()

    return (db(pred) - db(target)).pow(2).mean(dim=0).sqrt().mean().item()


def window_trend(history: Sequence[float], window: int = 50) -> Tuple[float, float]:
    """Mean loss over the first and last ``window`` steps."""
    window = max(1, min(window, len(history) // 2))
    return float(np.mean(history[:window])), float(np.mean(history[-window:]))


def save_wavs(result: DemoResult, out: Path) -> List[Path]:
    """Write the best prediction to ``out``, plus target and initial audio beside it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        (out, result.final_audio),
        (out.with_name(f"{out.stem}_target.wav"), result.target),
        (out.with_name(f"{out.stem}_initial.wav"), result.initial_audio),
    ]
    for path, audio in paths:
        # FLOAT subtype: avoid clipping if the prediction exceeds +/-1.
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
                        help="parameter initialization (default: npz for wav, perturbed for self)")
    parser.add_argument("--perturb", type=float, default=defaults.perturb,
                        help="uniform +/- perturbation for --init perturbed (normalized units)")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--pitch-lr-scale", type=float, default=defaults.pitch_lr_scale,
                        help="gp learning rate as a fraction of --lr")
    parser.add_argument("--lr-decay", action=argparse.BooleanOptionalAction,
                        default=defaults.lr_decay, help="cosine learning-rate decay")
    parser.add_argument("--l1-weight", type=float, default=defaults.l1_weight)
    parser.add_argument("--train-mapper", action=argparse.BooleanOptionalAction, default=None,
                        help="also optimize ParamToLPC (default: on for wav, off for self)")
    parser.add_argument("--freeze", type=parse_freeze, default={},
                        help="comma-separated params to hold fixed, optionally at a value: aa=0,fa")
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
          f"steps={cfg.steps} lr={cfg.lr:g} pitch_lr_scale={cfg.pitch_lr_scale:g} "
          f"train_mapper={cfg.resolved_train_mapper()} freeze={cfg.freeze or 'none'}")
    result = run(cfg)
    for line in result.log:
        print(f"  {line}")

    for path in save_wavs(result, cfg.out):
        print(f"  wrote {path}")

    first, last = window_trend(result.history)
    truth, start, best = result.truth_params, result.initial_params, result.params
    print(f"  loss: initial {result.initial_loss:.4f} -> best {result.final_loss:.4f} "
          f"at step {result.best_step} ({result.seconds:.1f}s)")
    print(f"  mean loss first/last 50 steps: {first:.4f} -> {last:.4f}")
    print(f"  log-spectral distance to target: "
          f"{log_spectral_distance(result.initial_audio, result.target):.1f} dB -> "
          f"{log_spectral_distance(result.final_audio, result.target):.1f} dB")
    print(f"  median gp: {median_pitch_hz(best):.1f} Hz (npz params: {median_pitch_hz(truth):.1f} Hz)")
    if cfg.target_mode == "self":
        before, after = param_error(start, truth, result.frozen), param_error(best, truth, result.frozen)
        print(f"  param error vs truth (mean |delta|, unfrozen): {before:.4f} -> {after:.4f}")
        print(f"  pitch error vs truth: {pitch_error_cents(start, truth):.0f} -> "
              f"{pitch_error_cents(best, truth):.0f} cents")
        if after >= before:
            print("  NOTE: parameters moved away from the truth; the loss can fall without "
                  "recovering parameters")

    if result.diverged_at is not None:
        print(f"FAIL: loss became non-finite at step {result.diverged_at}", file=sys.stderr)
        return 1
    if not result.final_loss < result.initial_loss:
        print("FAIL: loss did not decrease below its initial value", file=sys.stderr)
        return 1
    print("PASS: loss decreased")
    return 0


if __name__ == "__main__":
    sys.exit(main())
