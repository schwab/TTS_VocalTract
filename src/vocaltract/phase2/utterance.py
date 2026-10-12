"""
Fit a whole recorded utterance with the differentiable pipeline, starting from level A.

    recording --level A (vocaltract.analysis.resynth)--> engine frames (9.8 ms)
              --engine_to_grid--> params (1, N, 13) on a 10 or 20 ms grid
    params --pipeline (engine source, FormantToLPC, unnormalized)--> voice
    level-A fa/ff --reference engine--> frication track (fixed)
    prediction = gain * (voice + fric_gain * frication) + noise floor

and gradient descent on the parameter frames, the mapper's bandwidths and
tilt, and the two gains, against the multi-scale log-spectral loss.

Design choices, found by measurement on the evaluation sentences:

- **The whole utterance in one pass.** A 15 s sentence at a 10 ms hop takes
  about 1.5 GB and 1.5 s per step on CPU, so the overlapping windows the
  Phase 2 plan suggested (to bound memory) are not needed yet. Fitting in one
  piece avoids stitching: no phase resets or cross-fades.
- **No per-frame gain normalization** (``normalize_gain=False``). With the
  formant mapper (one resonator per formant, 100 Hz initial bandwidth like
  the engine's CORDIC damping) the unnormalized pipeline follows the engine's
  output level frame by frame: on a level-A utterance its voice level
  differed from the engine's by a constant to within 0.7 dB (interquartile
  range). With normalization the spread was 6 dB. Level A calibrates ``ga``
  and ``aa`` against the engine's level, so this is what makes its
  amplitudes a good starting point.
- **Frication is level A's, rendered by the reference engine and held
  fixed** (scaled by one learned gain). The pipeline does not model ``fa``/
  ``ff`` yet (Phase 2 step 2.2); without it fricatives would vanish and the
  fit would be judged on a handicap. The engine is linear in its sources up
  to 32-bit wraparound, and level A backs its gain off until the render
  does not overflow, so the voice and frication renders add up exactly to
  the full render.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from scipy import signal

from ..analysis.resynth import HOP_S as ENGINE_HOP_S
from ..analysis.resynth import render_frames
from ..losses import MultiScaleSTFTLoss
from ..mapper import make_mapper
from ..phase1.overfit_demo import FrameParams, Recording
from ..pipeline import synthesize
from ..reference.engine import AA, FA, GA, SAMPLE_RATE as ENGINE_RATE

SAMPLE_RATE = 16000
NOISE_SEED = 0
WAVEFORM = "vocaltract_aliased"  # the engine's own pulse, at its native rate
LEVEL_RANGE_DB = 35.0  # frames within this of the loudest calibrate the engine/pipeline scale


@dataclass
class FitConfig:
    """Settings for one utterance fit."""

    hop: int = 160  # samples at 16 kHz: 160 = 10 ms, 320 = 20 ms
    steps: int = 300
    lr: float = 1e-3
    pitch_lr_scale: float = 0.1
    lr_decay: bool = True
    l1_weight: float = 0.1
    log_floor_db: Optional[float] = 80.0
    train_mapper: bool = True
    noise_floor: bool = True
    frication: bool = True
    fit_frication_gain: bool = True
    log_every: int = 50
    seed: int = 0


@dataclass
class FitResult:
    """Outcome of a fit. ``params``/``audio`` are from the best step."""

    params: torch.Tensor  # (1, N, 13) normalized
    initial_params: torch.Tensor
    audio: np.ndarray  # 16 kHz, same length as the target
    initial_audio: np.ndarray
    history: List[float]
    best_step: int
    bandwidths_hz: List[float]
    frication_gain: float  # final, relative to the level-A mix
    seconds: float
    model: "Utterance" = None  # the trained model (mapper and frication gain) from the best step
    log: List[str] = field(default_factory=list)


def to_rate(audio: np.ndarray, sr: int, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Polyphase resampling to ``rate`` (float64)."""
    audio = np.asarray(audio, dtype=np.float64)
    if sr == rate:
        return audio
    g = np.gcd(int(sr), int(rate))
    return signal.resample_poly(audio, rate // g, int(sr) // g)


def engine_to_grid(frames: np.ndarray, n_frames: int, hop: int,
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Level-A engine frames -> normalized params at the pipeline's frame centres.

    VocalTract moves every parameter linearly from its previous value to
    frame k's over frame k's interval, starting from all zeros, so frame k is
    reached at (k + 1) * 9.8 ms. That piecewise-linear trajectory is sampled
    at the centres (j + 0.5) * hop of the pipeline frames (the convention of
    ``source.upsample_frames``) and divided by 255.

    Returns:
        (n_frames, 13) float64 in [0, 1].
    """
    frames = np.asarray(frames, dtype=np.float64)
    knots_t = np.concatenate([[0.0], (np.arange(len(frames)) + 1) * ENGINE_HOP_S])
    knots_v = np.vstack([np.zeros(frames.shape[1]), frames])
    centres = (np.arange(n_frames) + 0.5) * hop / sample_rate
    grid = np.stack([np.interp(centres, knots_t, knots_v[:, i]) for i in range(frames.shape[1])], 1)
    return grid / 255.0


def render_engine(frames: np.ndarray, n_samples: int) -> np.ndarray:
    """Reference-engine render of level-A frames at 16 kHz (engine units / 2**31)."""
    audio = render_frames(frames).astype(np.float64) / 2**31
    audio = signal.resample_poly(audio, SAMPLE_RATE // 4000, ENGINE_RATE // 4000)
    return np.pad(audio, (0, max(0, n_samples - len(audio))))[:n_samples]


def split_sources(frames: np.ndarray) -> tuple:
    """(voice frames: aa and ga only, frication frames: fa and ff only)."""
    voice, fric = frames.copy(), frames.copy()
    voice[:, FA] = 0
    fric[:, [GA, AA]] = 0
    return voice, fric


def frame_levels_db(audio: np.ndarray, hop: int) -> np.ndarray:
    """Per-frame RMS level in dB."""
    n = len(audio) // hop
    power = np.square(audio[: n * hop]).reshape(n, hop).mean(axis=1)
    return 10 * np.log10(power + 1e-20)


def engine_scale(engine_voice: np.ndarray, pipeline_voice: np.ndarray, hop: int) -> float:
    """
    Engine units per pipeline unit for the same frames: the median per-frame
    level ratio over the frames within ``LEVEL_RANGE_DB`` of the loudest.
    """
    e, p = frame_levels_db(engine_voice, hop), frame_levels_db(pipeline_voice, hop)
    n = min(len(e), len(p))
    e, p = e[:n], p[:n]
    loud = e > e.max() - LEVEL_RANGE_DB
    if not loud.any():
        return 1.0
    return float(10 ** (np.median(e[loud] - p[loud]) / 20))


class Utterance:
    """The differentiable model of one utterance: pipeline voice plus fixed frication."""

    def __init__(self, frication: Optional[np.ndarray], cfg: FitConfig):
        self.cfg = cfg
        self.mapper = make_mapper("formant", ENGINE_RATE).requires_grad_(cfg.train_mapper)
        self.frication = None if frication is None else torch.from_numpy(frication).float()[None]
        self.log_fric_gain = torch.tensor(0.0, requires_grad=cfg.fit_frication_gain)

    def voice(self, params: torch.Tensor, seed: int = NOISE_SEED) -> torch.Tensor:
        """Pipeline voice (glottal pulse + aspiration through the formants); ``seed``: aspiration noise."""
        return synthesize(params, self.mapper, hop=self.cfg.hop, sample_rate=SAMPLE_RATE,
                          generator=torch.Generator().manual_seed(seed), normalize_gain=False,
                          waveform=WAVEFORM, render_rate=ENGINE_RATE)

    @torch.no_grad()
    def render(self, params: torch.Tensor, seed: int = NOISE_SEED) -> np.ndarray:
        """16 kHz audio of ``params`` (1, N, 13) as VocalTract would play it: no recording gain or floor."""
        return self(params, seed)[0].double().numpy()

    def __call__(self, params: torch.Tensor, seed: int = NOISE_SEED) -> torch.Tensor:
        out = self.voice(params, seed)
        if self.frication is not None:
            out = out + self.log_fric_gain.exp() * self.frication[..., : out.shape[-1]]
        return out


def fit(target: np.ndarray, init_params: np.ndarray, frication: Optional[np.ndarray] = None,
        cfg: Optional[FitConfig] = None) -> FitResult:
    """
    Fit parameter frames to a 16 kHz recording.

    Args:
        target: (T,) 16 kHz audio; trimmed to whole frames.
        init_params: (N, 13) normalized starting frames, N = T // hop.
        frication: (T,) fixed frication track in pipeline units, or None.
        cfg: settings.
    """
    cfg = cfg or FitConfig()
    torch.manual_seed(cfg.seed)
    n = len(init_params)
    target_t = torch.from_numpy(np.asarray(target[: n * cfg.hop], dtype=np.float32))[None]
    if target_t.shape[-1] < n * cfg.hop:
        raise ValueError(f"target has {target_t.shape[-1]} samples, need {n * cfg.hop}")
    start = torch.from_numpy(np.asarray(init_params, dtype=np.float32))[None]
    frames = FrameParams(start)
    model = Utterance(frication if cfg.frication else None, cfg)
    recording = Recording(target_t, cfg.noise_floor)

    groups = [{"params": [frames.rest], "lr": cfg.lr},
              {"params": [frames.pitch], "lr": cfg.lr * cfg.pitch_lr_scale},
              {"params": recording.parameters(), "lr": Recording.LR}]
    if cfg.train_mapper:
        groups.append({"params": list(model.mapper.parameters()), "lr": cfg.lr})
    if cfg.fit_frication_gain and model.frication is not None:
        groups.append({"params": [model.log_fric_gain], "lr": Recording.LR})
    optimizer = torch.optim.Adam(groups)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(cfg.steps, 1))
                 if cfg.lr_decay else None)
    loss_fn = MultiScaleSTFTLoss(log_floor_db=cfg.log_floor_db)

    history, log = [], []
    best_step, best_params, best_audio, initial_audio = 0, start, None, None
    best_state = None
    begin = time.perf_counter()
    for step in range(cfg.steps + 1):
        params = frames.assemble()
        pred = recording(model(params))
        loss = loss_fn(pred, target_t) + cfg.l1_weight * (pred - target_t).abs().mean()
        if not torch.isfinite(loss):
            log.append(f"step {step:4d}  loss is non-finite, stopping")
            break
        history.append(loss.item())
        if initial_audio is None:
            initial_audio = pred.detach()
        if history[-1] <= history[best_step] or best_audio is None:
            best_step, best_params, best_audio = step, params.detach().clone(), pred.detach()
            best_state = (copy.deepcopy(model.mapper.state_dict()), model.log_fric_gain.item())
        if step % cfg.log_every == 0 or step == cfg.steps:
            log.append(f"step {step:4d}  loss {loss.item():.4f}  ({time.perf_counter() - begin:.0f}s)")
        if step == cfg.steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        frames.clamp_()

    model.mapper.load_state_dict(best_state[0])
    with torch.no_grad():
        model.log_fric_gain.fill_(best_state[1])
    return FitResult(
        params=best_params, initial_params=start,
        audio=best_audio[0].double().numpy(), initial_audio=initial_audio[0].double().numpy(),
        history=history, best_step=best_step,
        bandwidths_hz=model.mapper.bandwidths_hz().detach().tolist(),
        frication_gain=float(np.exp(best_state[1])), seconds=time.perf_counter() - begin,
        model=model, log=log,
    )


def prepare(frames: np.ndarray, n_samples: int, hop: int) -> tuple:
    """
    Level-A frames -> (init params (N, 13), frication track in pipeline units, engine render).

    The frication track is the engine's render of the ``fa``/``ff`` part
    alone, divided by the engine/pipeline level scale measured on the voice
    part, so that ``voice(init) + frication`` mixes like level A.
    """
    n = n_samples // hop
    init = engine_to_grid(frames, n, hop)
    voice_frames, fric_frames = split_sources(frames)
    engine_voice = render_engine(voice_frames, n * hop)
    engine_fric = render_engine(fric_frames, n * hop)
    model = Utterance(None, FitConfig(hop=hop))
    with torch.no_grad():
        pipeline_voice = model.voice(torch.from_numpy(init).float()[None])[0].double().numpy()
    scale = engine_scale(engine_voice, pipeline_voice, hop)
    return init, engine_fric / scale, engine_voice + engine_fric

