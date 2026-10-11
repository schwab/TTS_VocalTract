"""
Phase 1 "hello world": overfit one 1-second clip by optimizing parameter frames.

The free variable is the (1, 50, 13) parameter tensor and, by default for WAV
targets, the mapper's few learned values (formant bandwidths and spectral tilt
for the default ``FormantToLPC``; ``--mapper mlp`` selects the original MLP). Each step synthesizes audio with
:func:`vocaltract.pipeline.synthesize` (harmonic source, gain-normalized LPC
filter) and minimizes ``MultiScaleSTFTLoss + l1_weight * L1(waveform)`` with
Adam.

Targets come from ``assets/phase1`` (see its README):

- ``--target-mode wav`` (default, Step 7 option b): load the target WAV.
- ``--target-mode self`` (option a): render the target with this same pipeline
  from the target's ``.npz`` parameters and a frozen fixed-seed mapper, start
  from perturbed parameters, and report how far optimization moves them back
  toward the truth. This is a pass/fail test: exit 1 unless the error of the
  parameters that affect the audio falls. Pitch starts at the truth unless
  ``--perturb-pitch``: the spectral loss only pulls pitch back from within
  ~6 cents (harmonic k moves k times as far as f0), so a perturbed pitch
  masks whether the formants are recovered.

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
- The default mapper places one resonator at each of f1-f4 (as VocalTract
  does), so the formant parameters are identifiable. The MLP mapper
  (``--mapper mlp``) is non-identifiable: on vt_glide_ah_ee it held one
  compromise vowel while the target glided from /a/ to /i/.
- The mapper is trained by default on WAV targets (bandwidths and tilt are
  not given by the .npz params).
- The default source (``--source engine``) emulates VocalTract: its own
  glottal pulse, unfiltered, rendered at 20 kHz through the complex
  resonator structure, then resampled to 16 kHz exactly as the assets were.
  At the true parameters of vt_vowel_ah this matches the target spectrum to
  0.0 dB from 100 Hz to 4 kHz; the sawtooth/all-pole model was off by 3-40 dB
  and the optimizer moved formants to compensate.
- WAV targets get a learned output gain and noise floor (:class:`Recording`)
  and the log-spectral loss ignores content 80 dB below the target's peak.
  Without them the loss minimum is not at the true parameters.
- The summary reports formant error measured from the audio
  (:mod:`vocaltract.formants`), because spectral distance alone does not
  show whether formants move.
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
from vocaltract.formants import formant_error_hz
from vocaltract.mapper import make_mapper
from vocaltract.params import PARAM_NAMES, denormalize
from vocaltract.pipeline import synthesize as pipeline_synthesize

SAMPLE_RATE = 16000
ENGINE_RATE = 20000  # VocalTract's native sample rate
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
    mapper: str = "formant"
    source: str = "engine"
    init: Optional[str] = None  # None -> "npz" for wav targets, "perturbed" for self targets
    perturb: float = 0.05
    perturb_pitch: bool = False
    steps: int = 500
    lr: float = 1e-3
    pitch_lr_scale: float = 0.1
    lr_decay: bool = True
    l1_weight: float = 0.1
    log_floor_db: Optional[float] = 80.0
    train_mapper: Optional[bool] = None  # None -> True for wav targets, False for self targets
    noise_floor: Optional[bool] = None  # None -> True for wav targets, False for self targets
    output_gain: Optional[bool] = None  # None -> True for wav targets, False for self targets
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

    def resolved_noise_floor(self) -> bool:
        if self.noise_floor is not None:
            return self.noise_floor
        return self.target_mode == "wav"

    def resolved_output_gain(self) -> bool:
        if self.output_gain is not None:
            return self.output_gain
        return self.target_mode == "wav"

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
    active: List[str] = field(default_factory=lambda: list(PARAM_NAMES))
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


# --source choices: (waveform, render rate). "engine" emulates VocalTract: its own
# glottal pulse, unfiltered, at its native 20 kHz, then resampled to 16 kHz the
# way the assets were made. The others render at 16 kHz with band-limited waveforms.
SOURCES = {
    "engine": ("vocaltract_aliased", ENGINE_RATE),
    "vocaltract": ("vocaltract", SAMPLE_RATE),
    "sawtooth": ("sawtooth", SAMPLE_RATE),
}


def build_mapper(kind: str, source: str = "engine") -> torch.nn.Module:
    """A mapper for the render rate that ``source`` uses."""
    return make_mapper(kind, SOURCES[source][1])


def synthesize(
    params: torch.Tensor, mapper: torch.nn.Module, source: str = "engine"
) -> torch.Tensor:
    """params (B, T_frames, 13) -> 16 kHz audio (B, T_frames * HOP) with fixed-seed noise."""
    waveform, render_rate = SOURCES[source]
    noise = torch.Generator().manual_seed(NOISE_SEED)
    return pipeline_synthesize(params, mapper, hop=HOP, sample_rate=SAMPLE_RATE, generator=noise,
                               waveform=waveform, render_rate=render_rate)


class Recording:
    """
    How the target was recorded, outside the 13 parameters: an output gain and
    (optionally) an additive white noise floor, both learned.

    - **Gain.** The target WAVs are peak-normalized (to 0.5), so their level
      says nothing about ``ga``. Without a free gain the optimizer can only
      raise loudness by distorting the filter (wider bandwidths, spectral
      tilt, shifted formants). On vt_vowel_ah the true parameters are 3.6 dB
      too quiet even at ``ga`` = 1. The gain starts at the RMS ratio of
      target to initial prediction.
    - **Noise floor.** The VocalTract targets carry the engine's fixed-point
      rounding noise, a floor ~80 dB below the formant peaks that fills 5-8
      kHz; recordings have one too. Without it the optimizer fakes one with
      aspiration noise.

    Both are learned in log units with their own step size (LR), since they
    may need to move by orders of magnitude.
    """

    INITIAL_FLOOR = 1e-6
    LR = 0.05

    def __init__(self, target: torch.Tensor, noise_floor: bool, seed: int = NOISE_SEED + 1):
        self.target_rms = target.pow(2).mean().sqrt().item()
        self.log_gain = torch.tensor(0.0, requires_grad=True)
        self.calibrated = False
        self.noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed))
        self.log_floor = (torch.tensor(float(np.log(self.INITIAL_FLOOR)), requires_grad=True)
                          if noise_floor else None)

    def parameters(self) -> List[torch.Tensor]:
        return [self.log_gain] + ([self.log_floor] if self.log_floor is not None else [])

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if not self.calibrated:  # first call: start at the RMS ratio
            rms = audio.detach().pow(2).mean().sqrt().clamp_min(1e-12).item()
            with torch.no_grad():
                self.log_gain.fill_(float(np.log(self.target_rms / rms)))
            self.calibrated = True
        out = audio * self.log_gain.exp()
        if self.log_floor is not None:
            out = out + self.log_floor.exp() * self.noise
        return out

    @property
    def gain(self) -> float:
        return self.log_gain.exp().item()


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
    mapper = build_mapper(cfg.mapper, cfg.source).requires_grad_(train_mapper)

    wav, truth = load_asset(cfg.target, cfg.assets_dir)
    if cfg.target_mode == "wav":
        target = wav
    elif cfg.target_mode == "self":
        with torch.no_grad():
            target = synthesize(truth, mapper, cfg.source)
    else:
        raise ValueError(f"unknown target mode {cfg.target_mode!r}")

    start = initial_params(init, truth, cfg.perturb, cfg.seed)
    if init == "perturbed" and not cfg.perturb_pitch:
        start[..., GP_INDEX] = truth[..., GP_INDEX]
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
    recording = None
    if cfg.resolved_output_gain() or cfg.resolved_noise_floor():
        recording = Recording(target, cfg.resolved_noise_floor())
        if not cfg.resolved_output_gain():
            recording.calibrated = True  # gain stays at 1
            recording.log_gain.requires_grad_(False)
        groups.append({"params": [t for t in recording.parameters() if t.requires_grad],
                       "lr": Recording.LR})
    optimizer = torch.optim.Adam(groups)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(cfg.steps, 1))
        if cfg.lr_decay else None
    )
    loss_fn = MultiScaleSTFTLoss(log_floor_db=cfg.log_floor_db)
    frozen = list(cfg.freeze)

    history, log = [], []
    best_step, best_params, best_audio, initial_audio = 0, start, None, None
    diverged_at = None
    begin = time.perf_counter()
    for step in range(cfg.steps + 1):
        params = frames.assemble()
        pred = synthesize(params, mapper, cfg.source)
        if recording is not None:
            pred = recording(pred)
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
        active=active_params(mapper),
        diverged_at=diverged_at,
        seconds=time.perf_counter() - begin,
        log=log,
    )


def active_params(mapper: torch.nn.Module) -> List[str]:
    """Parameters that affect the audio: the source's (aa, ga, gp) plus the mapper's inputs."""
    used = {"aa", "ga", "gp", *getattr(mapper, "input_names", PARAM_NAMES)}
    return [name for name in PARAM_NAMES if name in used]


def formant_param_error_hz(params: torch.Tensor, truth: torch.Tensor) -> List[float]:
    """Mean absolute f1-f4 parameter error in Hz."""
    pred, true = denormalize(params), denormalize(truth)
    return [(pred[n] - true[n]).abs().mean().item() for n in ("f1", "f2", "f3", "f4")]


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
    parser.add_argument("--perturb-pitch", action=argparse.BooleanOptionalAction,
                        default=defaults.perturb_pitch,
                        help="also perturb gp for --init perturbed (default: pitch starts at truth)")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--pitch-lr-scale", type=float, default=defaults.pitch_lr_scale,
                        help="gp learning rate as a fraction of --lr")
    parser.add_argument("--lr-decay", action=argparse.BooleanOptionalAction,
                        default=defaults.lr_decay, help="cosine learning-rate decay")
    parser.add_argument("--l1-weight", type=float, default=defaults.l1_weight)
    parser.add_argument("--log-floor-db", type=lambda v: None if v == "none" else float(v),
                        default=defaults.log_floor_db,
                        help="dynamic range of the log-spectral loss below the target's peak "
                             "(default 80; 'none' for no floor)")
    parser.add_argument("--noise-floor", action=argparse.BooleanOptionalAction, default=None,
                        help="learn an additive output noise floor (default: on for wav)")
    parser.add_argument("--output-gain", action=argparse.BooleanOptionalAction, default=None,
                        help="learn an output gain (default: on for wav, whose level is arbitrary)")
    parser.add_argument("--mapper", choices=("formant", "mlp"), default=defaults.mapper,
                        help="formant: resonators at f1-f4 (default); mlp: original ParamToLPC")
    parser.add_argument("--source", choices=tuple(SOURCES), default=defaults.source,
                        help="engine: VocalTract pulse at 20 kHz, resampled (default); "
                             "vocaltract: band-limited pulse at 16 kHz; sawtooth")
    parser.add_argument("--train-mapper", action=argparse.BooleanOptionalAction, default=None,
                        help="also optimize the mapper (default: on for wav, off for self)")
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
    recovered = True
    try:
        init = cfg.resolved_init()
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print(f"Phase 1 overfit: target={cfg.target} mode={cfg.target_mode} mapper={cfg.mapper} source={cfg.source} "
          f"init={init} "
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
    before_hz = formant_error_hz(result.initial_audio, result.target)
    after_hz = formant_error_hz(result.final_audio, result.target)
    print(f"  formant error measured from audio, F1/F2/F3: "
          f"{'/'.join(f'{v:.0f}' for v in before_hz)} -> {'/'.join(f'{v:.0f}' for v in after_hz)} Hz")
    if cfg.target_mode == "self":
        skip = set(result.frozen) | (set(PARAM_NAMES) - set(result.active))
        before, after = param_error(start, truth, skip), param_error(best, truth, skip)
        print(f"  param error vs truth (mean |delta| over {', '.join(n for n in result.active if n not in skip)}): "
              f"{before:.4f} -> {after:.4f}")
        print(f"  f1-f4 param error vs truth: "
              f"{'/'.join(f'{v:.0f}' for v in formant_param_error_hz(start, truth))} -> "
              f"{'/'.join(f'{v:.0f}' for v in formant_param_error_hz(best, truth))} Hz")
        print(f"  pitch error vs truth: {pitch_error_cents(start, truth):.0f} -> "
              f"{pitch_error_cents(best, truth):.0f} cents")
        recovered = after < before

    if result.diverged_at is not None:
        print(f"FAIL: loss became non-finite at step {result.diverged_at}", file=sys.stderr)
        return 1
    if not result.final_loss < result.initial_loss:
        print("FAIL: loss did not decrease below its initial value", file=sys.stderr)
        return 1
    if not recovered:
        print("FAIL: parameters did not move toward the truth (the loss can fall "
              "without recovering parameters)", file=sys.stderr)
        return 1
    print("PASS: loss decreased" + (" and parameters moved toward the truth"
                                    if cfg.target_mode == "self" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
