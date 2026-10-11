# Phase 1 Task Document — Minimal Differentiable LPC Pipeline

**Repo:** `schwab/TTS_VocalTract`
**Phase:** 1 of 7
**Objective:** Build and verify the smallest end-to-end loop: `random parameter frames → LPC coefficients → differentiable filter → audio → gradient flows back to parameters`.

This phase does **not** involve real speech, neural encoders, or glottal modeling. It only proves the differentiable LPC core works, is stable, and is trainable.

---

## Phase 1 Goals

By the end of Phase 1, the repo must demonstrate:

1. A **parameter frame tensor** of shape `(batch, time, 13)` can be converted into **LPC coefficients** of shape `(batch, time, order)`.
2. A **differentiable LPC filter** synthesizes audio from those coefficients.
3. **Gradients flow** from a scalar loss back through the filter to the parameter tensor.
4. A **minimal training loop** can overfit a single synthetic target audio clip by adjusting only the input parameters.
5. All code is **CPU-runnable** and covered by a smoke test.

Success criterion: `pytest tests/` passes and `python -m vocaltract.phase1.overfit_demo` produces a WAV file that audibly converges toward the target over training.

---

## Environment & Dependencies

```bash
python >= 3.10
torch >= 2.1
torchlpc >= 0.2        # pip install torchlpc
numpy
scipy                  # for WAV I/O and STFT reference
soundfile
pytest
tqdm
```

Create `requirements.txt` with pinned versions. No CUDA required.

---

## Repository Layout (Phase 1 subset)

```
TTS_VocalTract/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── vocaltract/
│       ├── __init__.py
│       ├── params.py            # 13-param schema & defaults
│       ├── mapper.py            # params -> LPC coefficients
│       ├── filter.py            # differentiable LPC wrapper
│       ├── source.py            # placeholder excitation (Phase 1 stub)
│       ├── pipeline.py          # params -> mapper -> source -> filter -> audio
│       ├── losses.py            # multi-scale STFT loss
│       └── phase1/
│           ├── __init__.py
│           ├── smoke_test.py    # gradient flow check
│           └── overfit_demo.py  # training loop on one clip
├── assets/phase1/               # target WAV + ground-truth .npz params (see its README)
└── tests/
    ├── test_params.py
    ├── test_mapper.py
    ├── test_filter.py
    ├── test_source.py
    ├── test_losses.py
    ├── test_pipeline.py
    ├── test_gradients.py
    ├── test_smoke.py
    └── test_overfit_demo.py
```

---

## Step-by-Step Tasks

### Step 1 — Define the parameter schema

**File:** `src/vocaltract/params.py`

**Goal:** Establish the canonical 13-parameter frame as a documented, typed structure, reconciled against the original Propeller VocalTract.spin source code.

**Specification:**
- Define a dataclass `VocalParams` with 13 named fields matching the original VocalTract encoding.
- The original VocalTract uses byte-range parameters (0..255 for amplitudes, discrete units for frequencies). For differentiable training, we normalize to [0,1] range.

| Index | Name | Original Unit | Normalized Range | Meaning |
|---|---|---|---|---|
| 0 | `aa` | 0..255 | [0, 1] | Aspiration amplitude (breath volume: silent..loud, linear) |
| 1 | `ga` | 0..255 | [0, 1] | Glottal amplitude (voice volume: silent..loud, linear) |
| 2 | `gp` | 1/48 octave | [0, 1] | Glottal pitch (100 ≡ 110.00Hz, musical note A2) |
| 3 | `vp` | 1/48 octave | [0, 1] | Vibrato pitch (48 ≡ ± 1/2 octave swing) |
| 4 | `vr` | 0.0763 Hz | [0, 1] | Vibrato rate (52 ≡ 4 Hz) |
| 5 | `f1` | 19.53 Hz | [0, 1] | Formant 1 frequency (40 ≡ 781 Hz) |
| 6 | `f2` | 19.53 Hz | [0, 1] | Formant 2 frequency (56 ≡ 1094 Hz) |
| 7 | `f3` | 19.53 Hz | [0, 1] | Formant 3 frequency (128 ≡ 2500 Hz) |
| 8 | `f4` | 19.53 Hz | [0, 1] | Formant 4 frequency (179 ≡ 3496 Hz) |
| 9 | `na` | 0..255 | [0, 1] | Nasal amplitude (anti-resonator level: off..on, linear) |
| 10 | `nf` | 19.53 Hz | [0, 1] | Nasal frequency (anti-resonator frequency: 102 ≡ 1992 Hz) |
| 11 | `fa` | 0..255 | [0, 1] | Frication amplitude (white noise volume: silent..loud, linear) |
| 12 | `ff` | 39.06 Hz | [0, 1] | Frication frequency (60 ≡ 2344 Hz, "Sh" sound) |

- Provide `PARAM_NAMES: list[str]` and `PARAM_DIM = 13`.
- Provide `denormalize(params: Tensor) -> dict[str, Tensor]` converting [0,1] values to physical units using the original VocalTract scaling factors.
- Provide `normalize(physical_values: dict) -> Tensor` for the inverse operation.

**Physical unit conversions (original VocalTract):**
- Amplitude parameters (`aa`, `ga`, `na`, `fa`): raw value / 255
- Pitch/rate parameters (`gp`, `vp`, `vr`, `f1-f4`, `nf`): raw_value * scale_factor
  - `gp`: (value - 100) maps to log-frequency around 110Hz
  - `vp`: (value - 48) maps to ±1/2 octave swing from base pitch
  - `vr`: value * 0.0763 Hz
  - `f1-f4`: value * 19.53 Hz
  - `nf`: value * 19.53 Hz
  - `ff`: value * 39.06 Hz

**Expected Output:**
- `from vocaltract.params import PARAM_DIM, PARAM_NAMES` works.
- `test_params.py` asserts `PARAM_DIM == 13`, all names are unique, and matches the original VocalTract ordering.
- Denormalization produces physically sensible values consistent with the Spin implementation.

---

### Step 2 — Implement the parameter → LPC coefficient mapper

**File:** `src/vocaltract/mapper.py`

**Goal:** A small, trainable module that converts a `(B, T, 13)` tensor into LPC coefficients `(B, T, order)` while guaranteeing filter stability.

**Specification:**
- Class `ParamToLPC(nn.Module)` with `__init__(self, order: int = 10, hidden: int = 64, gamma: float = 0.99)`.
- Architecture: `Linear(13, hidden) → GELU → Linear(hidden, hidden) → GELU → Linear(hidden, order)`.
- Output is interpreted as **reflection coefficients (PARCOR)** via `0.99 * tanh(x)`, i.e. `|k| ≤ 0.99`.
- Convert reflection coefficients → LPC coefficients using the **Levinson–Durbin recursion**, implemented differentiably in pure PyTorch (loop over order, vectorized over batch/time). The recursion and the returned coefficients are **float64**.
- Apply **bandwidth expansion** `a_i ← a_i · γ^i` (γ = 0.99): every pole radius is scaled by γ, so the narrowest resonance is ~51 Hz wide at 16 kHz.
- Provide `forward(params: Tensor) -> Tensor` returning float64 LPC coefficients `(B, T, order)`.
- Include a `to_reflection` helper and the inverse for tests.

**Why reflection coefficients:** guarantees pole stability by construction, and is a well-trodden path in the GOLF codebase.

**Why the cap, float64 and γ (revision):** "stable by construction" holds in exact arithmetic only. Measured during Step 7:
- `tanh` saturates to exactly ±1 in float32 at large pre-activations; `|k| = 1` puts a repeated root on the unit circle.
- Direct-form coefficients rounded to float32 realize an unstable filter once `|k|` exceeds ~0.9; float64 is exact at 0.99.
- Without γ, poles can sit arbitrarily close to the unit circle and the filter gain explodes; training the mapper at lr ≥ 1e-2 produced NaN. With all three fixes it trains without NaN up to lr 1e-1.

**Gain normalization** (in `pipeline.py`, not the mapper): the excitation is multiplied per frame by `min_ω |A(e^{jω})|` (`filter.min_response`), so the filter's peak gain is 1 and output level follows `ga`. Multiplying by the minimum rather than dividing by the peak keeps the gradient bounded.

**Expected Output:**
- `mapper = ParamToLPC(); A = mapper(torch.rand(2, 50, 13))` returns shape `(2, 50, 10)`.
- All poles lie strictly inside the unit circle, with radius ≤ γ (verified in `test_mapper.py` by checking the roots of the LPC polynomial for a batch of random inputs, including saturated ones).
- Gradients flow through `mapper` — `A.sum().backward()` populates `mapper` parameter grads.

---

### Step 3 — Wrap `torchlpc` as a differentiable filter module

**File:** `src/vocaltract/filter.py`

**Goal:** A thin, well-documented wrapper around `torchlpc.sample_wise_lpc` that handles shapes, initial conditions, and provides a clean API.

**Specification:**
- Class `LPCFilter(nn.Module)` with `forward(self, excitation: Tensor, coeffs: Tensor) -> Tensor`.
- `excitation`: `(B, T)` — the source signal.
- `coeffs`: `(B, T, order)` — time-varying LPC coefficients.
- Inside, call `torchlpc.sample_wise_lpc(excitation, coeffs)`.
- `torchlpc` uses the `1 + Σ A_i z^{-i}` sign convention; the mapper uses `1 − Σ a_i z^{-i}`, so the wrapper negates the coefficients.
- Filter at the higher precision of the two inputs (float64 when the mapper's coefficients are float64); return the excitation's dtype.
- Zero initial state for Phase 1.
- Add `@torch.no_grad()` helper `analyze(audio, order)` using `torchlpc.lpc` for reference comparison only.
- Docstring must explicitly note that `torchlpc` uses efficient custom kernels and provides gradients through a reverse filtering operation.

**Expected Output:**
- `LPCFilter()(torch.randn(2, 16000), torch.randn(2, 16000, 10) * 0.01)` returns `(2, 16000)`.
- `test_filter.py` checks output shape, dtype, and that a small-coefficient input yields near-identity filtering.

---

### Step 4 — Placeholder excitation source

**File:** `src/vocaltract/source.py`

**Goal:** A differentiable but intentionally simple excitation generator for Phase 1. Real glottal modeling is Phase 2.

**Specification:**
- Function `make_excitation(params: Tensor, sample_rate: int = 16000, hop: int = 320) -> Tensor`.
- Input: `(B, T_frames, 13)`; output: `(B, T_frames * hop)`.
- For Phase 1: generate a **band-limited harmonic source** at the frequency given by `denormalize(params)["gp"]`, scaled by `ga`, plus a white-noise component scaled by `aa`.
  - Harmonic sum `(2/π) Σ_k sin(k φ) / k` (a band-limited sawtooth, 1/k spectral tilt) with every harmonic up to Nyquist.
  - Each harmonic fades out linearly over the last 500 Hz below Nyquist, so harmonics enter and leave smoothly as pitch changes and nothing aliases.
  - `max_harmonics=1` reproduces the original single-sine source.
- Use `torch.cumsum` on the instantaneous frequency to keep phase continuity (wrapped to [0, 1) cycles), and `torch.sin` for the carriers — both differentiable.
- Interpolate frame-level parameters to sample-level using `F.interpolate(mode="linear")`.
- Accept an optional `torch.Generator` for the noise so synthesis is repeatable.

**Why harmonics (revision):** a pure sine has energy only at f0. The all-pole filter can only reshape energy that is present, so a sine-driven model could not produce the harmonic series of any target; the optimizer substituted aspiration noise instead.

**Expected Output:**
- Output shape `(B, T_frames * hop)`.
- No NaN/Inf for random inputs in [0, 1].
- Gradients flow from output to `params`.

---

### Step 5 — Multi-scale STFT loss

**File:** `src/vocaltract/losses.py`

**Goal:** A standard spectral loss used for Phase 1 training.

**Specification:**
- Class `MultiScaleSTFTLoss(nn.Module)` with `__init__(self, ffts=(512, 1024, 2048), hops=None, win_lengths=None)`.
- For each scale: compute magnitude STFT, then combine **spectral convergence** and **log-magnitude L1** losses.
- Return the mean across scales.
- Use `torch.stft` with `return_complex=True`, and always detach neither input nor target.

**Expected Output:**
- `loss = MultiScaleSTFTLoss()(pred, target)` returns a scalar tensor with `requires_grad=True` when `pred` does.
- `test_gradients.py` asserts the loss decreases after a single optimizer step on a toy problem.

---

### Step 6 — Gradient flow smoke test

**File:** `src/vocaltract/phase1/smoke_test.py`

**Goal:** Prove the whole chain is differentiable end-to-end.

**Specification:**
- Build `params = torch.rand(1, 50, 13, requires_grad=True)`.
- Run `params → mapper → source → filter → audio` via `vocaltract.pipeline.synthesize_trace` (the same chain the demo trains through).
- Compute `loss = audio.pow(2).mean()`.
- Call `loss.backward()`.
- Assert `params.grad is not None` and `params.grad.abs().sum() > 0`.
- Print a short summary: shapes at each stage, `loss.item()`, gradient norm.
- Exit non-zero on failure so it can be used in CI.

**Expected Output:**
- Running `python -m vocaltract.phase1.smoke_test` prints the shape trace and gradient norm, exits 0.

---

### Step 7 — Overfit demo on a single clip

**File:** `src/vocaltract/phase1/overfit_demo.py`

**Goal:** Train the parameter tensor (and mapper) to reconstruct a fixed target audio clip. This is the Phase 1 “hello world”.

**Specification:**
- Load or synthesize a target: use a 1-second, 16 kHz clip from `assets/phase1` (`--target`, default `vt_vowel_ah`). Either:
  - (a) `--target-mode self`: synthesize the target with the same pipeline from the clip's ground-truth `.npz` params and a frozen fixed-seed mapper; start from the truth perturbed by `--perturb` (default 0.05) and report parameter and pitch error before/after, or
  - (b) `--target-mode wav` (default): load the target WAV; start from its `.npz` params.
- Trainable objects: `params` (the free variable) and `mapper` (`--train-mapper/--no-train-mapper`; default on for WAV targets, off for self targets).
- `params` are optimized directly and clamped to [0, 1] after each step (projected gradient descent). A sigmoid parameterization cannot reach the many exact zeros in the targets.
- `--freeze name[=value],...` holds parameters fixed, e.g. `--freeze aa=0` removes aspiration noise.
- Optimizer: `Adam(lr=1e-3)`, with pitch (`gp`) at `lr × 0.1` (`--pitch-lr-scale`) and cosine learning-rate decay (`--no-lr-decay` to disable). The best step is kept.
- Loss: `MultiScaleSTFTLoss` + `0.1 ×` L1 on the waveform. Noise uses a fixed seed every step.
- Loop for a configurable number of steps (default 500), logging loss every 50 steps.
- Save the best prediction to `out/phase1_overfit.wav`, plus `_target.wav` and `_initial.wav` for listening.
- Print initial and best loss, log-spectral distance (LSD) to the target before/after, and median pitch. Exit 0 if the loss decreased, 1 if not or on divergence, 2 on a bad configuration.

**Expected Output:**
- `python -m vocaltract.phase1.overfit_demo` runs to completion in under 2 minutes on CPU (16–32 s measured).
- `out/phase1_overfit.wav` is written and its STFT visually resembles the target (manual check).
- Loss curve is decreasing on average (windowed trend); LSD falls to under 0.6× its starting value; median pitch ends within 2% of the target's.

**Measured results** (default settings, LSD in dB, lower is better):

| Target | Start | After 500 steps |
|---|---|---|
| synthetic_vowel_ah | 37.3 | 12.1 |
| vt_vowel_ah | 35.8 | 15.7 |
| vt_glide_ah_ee | 39.4 | 13.6 |
| vt_sung_oo_vibrato | 33.7 | 13.7 |
| vt_seven | 36.3 | 11.8 |
| speech_male | 26.3 | 12.7 |
| speech_female | 27.6 | 10.5 |

Pitch ends within 0.8 Hz of the target on every clip.

---

### Step 8 — Tests

**Files:** `tests/test_params.py`, `tests/test_mapper.py`, `tests/test_filter.py`, `tests/test_gradients.py`

**Goal:** Lock in the Phase 1 contract.

**Specification:**

- `test_params.py`
  - `PARAM_DIM == 13`, names unique, `denormalize` returns 13 keys.
- `test_mapper.py`
  - Output shape `(B, T, order)`.
  - Stability: for 100 random inputs, all pole radii `< 1.0` (allow `1e-6` tolerance).
  - Determinism with fixed seed.
- `test_filter.py`
  - Output shape matches input.
  - With near-zero coefficients, output ≈ input (within tolerance).
- `test_gradients.py`
  - End-to-end `params → audio → scalar loss` produces non-zero gradients on `params`.
  - One optimizer step reduces the loss on a fixed seed (with `gp` held fixed: pitch only enters through the cumulative phase, so its gradient is linear only over a tiny step).

**Expected Output:**
- `pytest -q` passes all tests in under 60 seconds on CPU.

---

### Step 9 — Update README with Phase 1 status

**File:** `README.md`

**Goal:** Reflect that Phase 1 is implemented.

**Specification:**
- Check the Phase 1 box in the roadmap.
- Add a `## Phase 1: Minimal Differentiable LPC Pipeline` section with:
  - A one-line description
  - Install instructions
  - The two commands to run (`smoke_test` and `overfit_demo`)
  - A short note that Phase 2 will replace the sine excitation with a real glottal model.

**Expected Output:**
- README renders correctly on GitHub, commands are copy-pasteable.

---

## Definition of Done (Phase 1)

- [ ] All nine steps complete.
- [ ] `pytest -q` passes.
- [ ] `python -m vocaltract.phase1.smoke_test` exits 0 and prints non-zero gradient norm.
- [ ] `python -m vocaltract.phase1.overfit_demo` writes `out/phase1_overfit.wav` and loss decreases.
- [ ] README updated.
- [ ] `requirements.txt` pinned and installable in a fresh venv.

---

## Known Limitations (found in Step 7)

- **Parameters are not identifiable through the MLP mapper.** In self-target mode the loss falls but parameter error grows (0.018 → 0.036 mean absolute error at `--perturb 0.05`); pitch error improves only from 32 to 27 cents. The mapper sees all 13 parameters, including ones Phase 1 does not model (`na`, `nf`, `ff`, `fa`), and a random network has no reason to place resonances at `f1`–`f4`, so many parameter settings sound alike. Fixing this needs a physically grounded mapper (formant frequencies → resonators), not more optimization.
- **Harmonic energy is still short of the targets** (about a third on `vt_vowel_ah`, measured before the final mapper and loss fixes). Likely the same cause: the learned filter shape, not the source.
- **Pitch is only locally learnable.** Phase is a running sum of frequency, so a pitch error compounds over the clip; from a neutral start, pitch gets stuck in local minima of the spectral loss. Demos start from the target's `.npz` params.

---

## Explicit Non-Goals for Phase 1

To keep scope tight, Phase 1 does **not** include:

- Real speech data or audio→parameter encoders
- A realistic glottal pulse (Phase 2)
- Formant tracking loss (Phase 4)
- Text input (Phase 5)
- CUDA / mixed precision
- Inference optimization or embedded export

Any of these appearing in Phase 1 PRs should be deferred.

---

## Notes for the Implementing Agent

- Prefer small, well-named functions over clever one-liners — this code will be read by others.
- Every module gets a module-level docstring explaining its role in the pipeline.
- Every public function gets type hints.
- Set `torch.manual_seed(0)` in demos and tests for reproducibility.
- If `torchlpc` is unavailable on the target platform, fall back to a naive differentiable IIR implementation in `filter.py` behind a feature flag, and document the performance caveat.
