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
│       ├── losses.py            # multi-scale STFT loss
│       └── phase1/
│           ├── __init__.py
│           ├── smoke_test.py    # gradient flow check
│           └── overfit_demo.py  # training loop on one clip
└── tests/
    ├── test_params.py
    ├── test_mapper.py
    ├── test_filter.py
    └── test_gradients.py
```

---

## Step-by-Step Tasks

### Step 1 — Define the parameter schema

**File:** `src/vocaltract/params.py`

**Goal:** Establish the canonical 13-parameter frame as a documented, typed structure.

**Specification:**
- Define a dataclass `VocalParams` with 13 named fields.
- Fields (adjust names to match the original Propeller VocalTract if you have the source; these are the working names):

| Index | Name | Range | Meaning |
|---|---|---|---|
| 0 | `aspiration` | [0, 1] | Breathiness / noise amplitude |
| 1 | `glottal_amp` | [0, 1] | Voicing amplitude |
| 2 | `glottal_pitch` | [0, 1] | Normalized F0 (maps to 50–400 Hz) |
| 3 | `f1` | [0, 1] | Formant 1 (maps to 200–1000 Hz) |
| 4 | `f2` | [0, 1] | Formant 2 (maps to 700–2500 Hz) |
| 5 | `f3` | [0, 1] | Formant 3 (maps to 2000–3500 Hz) |
| 6 | `f4` | [0, 1] | Formant 4 (maps to 3000–4500 Hz) |
| 7 | `b1` | [0, 1] | Formant 1 bandwidth |
| 8 | `b2` | [0, 1] | Formant 2 bandwidth |
| 9 | `b3` | [0, 1] | Formant 3 bandwidth |
| 10 | `fricative` | [0, 1] | Fricative noise amplitude |
| 11 | `nasal` | [0, 1] | Nasal anti-resonator depth |
| 12 | `reserved` | [0, 1] | Spare / tilt |

- Provide `PARAM_NAMES: list[str]` and `PARAM_DIM = 13`.
- Provide `denormalize(params: Tensor) -> dict[str, Tensor]` converting [0,1] values to physical units.
- Document that the schema is provisional and will be reconciled with the original VocalTract encoding in a later phase.

**Expected Output:**
- `from vocaltract.params import PARAM_DIM, PARAM_NAMES` works.
- `test_params.py` asserts `PARAM_DIM == 13` and that all names are unique.

---

### Step 2 — Implement the parameter → LPC coefficient mapper

**File:** `src/vocaltract/mapper.py`

**Goal:** A small, trainable module that converts a `(B, T, 13)` tensor into LPC coefficients `(B, T, order)` while guaranteeing filter stability.

**Specification:**
- Class `ParamToLPC(nn.Module)` with `__init__(self, order: int = 10, hidden: int = 64)`.
- Architecture: `Linear(13, hidden) → GELU → Linear(hidden, hidden) → GELU → Linear(hidden, order)`.
- Output is interpreted as **reflection coefficients (PARCOR)** in `(-1, 1)` via `tanh`.
- Convert reflection coefficients → LPC coefficients using the **Levinson–Durbin recursion**, implemented differentiably in pure PyTorch (loop over order, vectorized over batch/time).
- Provide `forward(params: Tensor) -> Tensor` returning LPC coefficients `(B, T, order)`.
- Include a `to_reflection` helper and the inverse for tests.

**Why reflection coefficients:** guarantees pole stability by construction, and is a well-trodden path in the GOLF codebase.

**Expected Output:**
- `mapper = ParamToLPC(); A = mapper(torch.rand(2, 50, 13))` returns shape `(2, 50, 10)`.
- All poles lie strictly inside the unit circle (verified in `test_mapper.py` by checking the roots of the LPC polynomial for a batch of random inputs).
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
- For Phase 1: generate a **sine-based periodic source** at the frequency given by `denormalize(params)["glottal_pitch"]`, plus a white-noise component scaled by `aspiration`.
- Use `torch.cumsum` on the instantaneous frequency to keep phase continuity, and `torch.sin` for the carrier — both differentiable.
- Interpolate frame-level parameters to sample-level using `F.interpolate(mode="linear")`.

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
- Run `params → mapper → source → filter → audio`.
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
- Load or synthesize a target: use a 1-second, 16 kHz clip. Either:
  - (a) synthesize a target with the same pipeline using fixed ground-truth params, or
  - (b) load any short WAV from `assets/` (create the folder, add a placeholder if empty).
- Trainable objects: `params` (the free variable) and `mapper` (optional; configurable via flag).
- Optimizer: `Adam(lr=1e-3)`.
- Loss: `MultiScaleSTFTLoss` + small L1 on the waveform.
- Loop for a configurable number of steps (default 500), logging loss every 50 steps.
- Save the final prediction to `out/phase1_overfit.wav`.
- Print initial and final loss; assert final < initial.

**Expected Output:**
- `python -m vocaltract.phase1.overfit_demo` runs to completion in under 2 minutes on CPU.
- `out/phase1_overfit.wav` is written and its STFT visually resembles the target (manual check).
- Loss curve is monotonically decreasing on average.

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
  - One optimizer step reduces the loss on a fixed seed.

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

---

Want me to also draft the **Phase 2 task document** (glottal source model) so you can see how Phase 1's stubs are meant to be replaced?
