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
│       ├── resample.py          # differentiable resample_poly (20 kHz engine rate -> 16 kHz)
│       ├── formants.py          # LPC formant measurement, for evaluation
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
    ├── test_formant_mapper.py
    ├── test_formants.py
    ├── test_resample.py
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

**Goal:** A module that converts a `(B, T, 13)` tensor into LPC coefficients `(B, T, order)` while guaranteeing filter stability, **such that the formant parameters `f1`–`f4` determine the formants** (revision 2, below).

**Specification — `FormantToLPC` (default, revision 2):**
- Class `FormantToLPC(nn.Module)` with `__init__(self, sample_rate=16000, bandwidths_hz=(100,)*4, tilt_poles=2, complex_resonators=True)`.
- Each of `f1`–`f4` (Hz via the VocalTract scale, 19.53 Hz/unit) places one resonator: pole pair `r e^{±jθ}`, `θ = 2π f / sample_rate`, `r = exp(−π · bandwidth / sample_rate)`. The four second-order sections and `tilt_poles` real poles are multiplied into one order-10 polynomial (float64), so the filter interface is unchanged. Only `f1`–`f4` reach the filter.
- Learned (6 values in total): one bandwidth per formant (initialized to VocalTract's ~100 Hz; floor 40 Hz; stored as `log(bw − 40)`), and the tilt poles (`0.9 · tanh`, initialized to 0).
- **VocalTract's complex resonators** (`complex_resonators=True`): the engine cascades complex one-pole resonators `s[n] = r e^{jθ}(s[n−1] + u[n])` and outputs the real part. That is exactly `N(z)/D(z)` with `D` the all-pole product above and an order-4 numerator `N(z) = Re[e^{jΣθ} · Π(1 − r e^{−jθ_i} z⁻¹)]`. `numerator(params)` returns `N` (B, T, 5); the pipeline applies it as a per-frame FIR (`filter.fir_frames`) before the all-pole filter. Verified sample-exact against a direct simulation of the cascade.
- Stability: every pole radius ≤ max(e^{−π·40/sr}, 0.9) < 1 for any input. Direct-form rounding with all four formants at 0 Hz moves the largest radius to 0.997 (still stable; the realized filter is tested by impulse-response decay).
- `make_mapper("formant" | "mlp", sample_rate)` builds either mapper.

**Why (revision 2):** with the MLP mapper, the parameters are not identifiable. On `vt_glide_ah_ee` it held one compromise vowel (F1/F2 ≈ 580/1470 Hz) for the whole clip while the target glided /a/ → /i/ (F2 1090 → 2270 Hz), yet scored a *better* log-spectral distance than the formant mapper. In self-consistent mode it reached near-zero loss with formants 25–69 Hz off; the formant mapper recovers them to 0–8 Hz.

**Specification — `ParamToLPC` (original MLP mapper, kept for comparison as `--mapper mlp`):**
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

**Gain normalization** (in `pipeline.py`, not the mapper): the excitation is multiplied per frame by `min_ω |A(e^{jω})| / |N(e^{jω})|` (`filter.min_response`; `N = 1` without a numerator), so the filter's peak gain is 1 and output level follows `ga`. Multiplying by the minimum rather than dividing by the peak keeps the gradient bounded.

**Expected Output:**
- `mapper = FormantToLPC(); A = mapper(torch.rand(2, 50, 13))` returns shape `(2, 50, 10)`; so does `ParamToLPC()`.
- `FormantToLPC` places poles at `f1`–`f4` with the configured bandwidths; moving `f2` moves only the second resonance; other parameters get zero gradient (`test_formant_mapper.py`).
- All poles lie strictly inside the unit circle, with radius ≤ γ (verified in `test_mapper.py` by checking the roots of the LPC polynomial for a batch of random inputs, including saturated ones).
- Gradients flow through `mapper` — `A.sum().backward()` populates `mapper` parameter grads.

#### Addendum — review of the Step 7 output (2026-10-11; full notes and reproduction in `docs/reviews/phase1_step7_review.md`)

*Status: addressed by revision 2 (formant mapper, engine source); kept for the record. Current state: `docs/reviews/phase1_completion_review.md`.*

**Finding: unused pole pairs get stuck and create resonances the target doesn't have.** Order 10 is five conjugate
pole pairs, and `vt_vowel_ah` has four formants. After `overfit_demo --freeze aa=0` (500 steps), the trained
filter's poles at the middle frame (frequency/bandwidth, Hz) were:

| Order | Poles | 4–6 kHz / 6–8 kHz level re peak (target −104 / −110 dB) | LSD |
|---|---|---|---|
| 10 (default) | 719/265, 1130/224, 2413/106, **4736/53**, 7955/117 | −74 / −79 | 15.6 |
| 8 | 747/227, 1357/443, 2672/182, 8000/51, 8000/wide | −98 / −103 | 12.4 |
| 12 | 0/wide, 728/276, 1138/227, 2411/73, 3286/57, **6765/55**, 8000/51 | −95 / −86 | 14.4 |

- At order 10 the spare pair sits at 4.7 kHz with the narrowest bandwidth γ allows. That is the stray band in the
  spectrogram, and the 4–8 kHz excess of +30 to +50 dB. **F4 (3.4 kHz) is not modelled at all.**
- Order 8 removes the high-band excess but loses F4 and misplaces F2. Order 12 finds F4 but parks spare poles at
  6.8 and 8 kHz. So the filter order isn't the fix. It's a local minimum: the loss does see the excess (see the
  Step 5 addendum), but gradient descent doesn't move a resonance across 1–2 kHz of spectrum.

**Recommendations, in order of preference:**
1. **Formant-parameterized filter (also fixes the identifiability limitation below).** Build the all-pole filter as
   a cascade of 2nd-order resonators, one per formant, with centre frequencies taken directly from `f1`–`f4`
   (VocalTract's own structure) and bandwidths from a small learned head or fixed at about 100 Hz. Add at most one
   extra real pole or low-order section for overall tilt. With no free resonances there's nowhere for a stray band
   to come from, and `f1`–`f4` mean what they say. Stability is a pole radius < 1 per section, which is trivially
   guaranteed.
2. **Interim, if the MLP mapper stays:** initialize the reflection coefficients beyond 2×(number of formants) near
   0, so spare sections start effectively disabled. Add a small penalty on pole pairs narrower than about 200 Hz
   that sit above F4 + 500 Hz.

Acceptance check for either: on `vt_vowel_ah`, no resonance above 4 kHz, and four resonances within 5% of
730/1090/2440/3400 Hz.

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

- `waveform="vocaltract"` (revision 2): the same harmonic machinery with the Fourier coefficients of **VocalTract's own glottal pulse**. From the engine's antilog/sine ROM arithmetic, with phase φ ∈ [0, 1) per period: `g(φ) = −cos(2π · 2^{16(φ−1)})`. It is flat at −1 for most of the period, then one fast pulse in the last ~20%.
- `waveform="vocaltract_aliased"`: `g(φ)` evaluated per sample, as the engine does. Not band-limited: rendered at the engine's 20 kHz, its aliasing is part of VocalTract's sound (it fills 4–8 kHz of the targets).
- `pipeline.synthesize(..., waveform=..., render_rate=20000)` runs source and filter at 20 kHz (mapper built for 20 kHz, 400 samples per frame) and resamples to 16 kHz with `resample.resample_poly`, a differentiable copy of `scipy.signal.resample_poly` (the assets were made with scipy's, so this matches them).

**Why the VocalTract pulse and engine rate (revision 2):** at the true parameters of `vt_vowel_ah`, the long-term spectrum of (complex resonators + `vocaltract_aliased` at 20 kHz + resample) matches the target to **0.0 dB from 100 Hz to 4 kHz**. With the sawtooth or an all-pole cascade, the mismatch is 3–40 dB depending on band, and the optimizer moved formants by 100+ Hz to compensate. Above 5 kHz the target is VocalTract's fixed-point rounding noise (a flat floor ~80 dB below the formant peaks).

**Why harmonics (revision):** a pure sine has energy only at f0. The all-pole filter can only reshape energy that is present, so a sine-driven model could not produce the harmonic series of any target; the optimizer substituted aspiration noise instead.

**Expected Output:**
- Output shape `(B, T_frames * hop)`.
- No NaN/Inf for random inputs in [0, 1].
- Gradients flow from output to `params`.

#### Addendum — review of the Step 7 output (2026-10-11; full notes and reproduction in `docs/reviews/phase1_step7_review.md`)

*Status: addressed by revision 2 (formant mapper, engine source); kept for the record. Current state: `docs/reviews/phase1_completion_review.md`.*

**Finding: the fixed 1/k source tilt is wrong for VocalTract, and the filter compensates by smearing F1/F2.**
In `vt_vowel_ah` the fundamental is about 25 dB below the strongest harmonic (h6, at F1). The spectrum *rises*
from 120 Hz to F1. The sawtooth source makes h1 the strongest harmonic. After training, harmonics 1–5 are still
+25, +16, +12, +10 and +6 dB too strong. The filter's only way to compensate is to widen F1/F2: 265/224 Hz pole
bandwidths, against about 100 Hz in VocalTract (r = 0.984 at 20 kHz). The F2–F3 valley (1.5–2.3 kHz) ends about
10 dB too shallow as a result.

**Recommendation:** make the source tilt learnable, independent of the filter.
- Simplest: `Σ_k sin(kφ) / k^α` with a learnable global α (init 1.0), plus a learnable one-pole high-pass on the
  excitation for the weak fundamental.
- Keep these as global (per-clip) learnables, not new frame parameters, so the 13-parameter schema is unchanged.
- This is the first piece of the Phase 2 glottal model. Phase 2 replaces it with a pulse shape, and the
  interface stays the same.

Acceptance check: on `vt_vowel_ah`, harmonics 1–5 within ±6 dB of the target (relative to each clip's strongest
harmonic), and F1/F2 bandwidths under 150 Hz.

---

### Step 5 — Multi-scale STFT loss

**File:** `src/vocaltract/losses.py`

**Goal:** A standard spectral loss used for Phase 1 training.

**Specification:**
- Class `MultiScaleSTFTLoss(nn.Module)` with `__init__(self, ffts=(512, 1024, 2048), hops=None, win_lengths=None)`.
- For each scale: compute magnitude STFT, then combine **spectral convergence** and **log-magnitude L1** losses.
- Return the mean across scales.
- Use `torch.stft` with `return_complex=True`, and always detach neither input nor target.

- Optional `log_floor_db` (revision 2): the log term compares `log(mag + floor)` with `floor` = the target's peak magnitude (per scale) minus `log_floor_db` dB. Content far below the floor (engine rounding noise, recording noise) stops dominating the loss; `None` (default) keeps the original behaviour.

**Expected Output:**
- `loss = MultiScaleSTFTLoss()(pred, target)` returns a scalar tensor with `requires_grad=True` when `pred` does.
- `test_gradients.py` asserts the loss decreases after a single optimizer step on a toy problem.

#### Addendum — review of the Step 7 output (2026-10-11; full notes and reproduction in `docs/reviews/phase1_step7_review.md`)

*Status: addressed by revision 2 (formant mapper, engine source); kept for the record. Current state: `docs/reviews/phase1_completion_review.md`.*

**Checked, no change needed.** It seemed possible that the log-magnitude term ignores the quiet high band because
of its floor. It doesn't: `EPS = 1e-7`, while the quietest 4–8 kHz STFT magnitudes of `vt_vowel_ah` are about
2e-4 (−102 dB re peak). The log-L1 term sees the +30 to +50 dB excess fine. The excess persists because of the
pole local minimum described in the Step 2 addendum, not the loss.

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
  - (a) `--target-mode self`: synthesize the target with the same pipeline from the clip's ground-truth `.npz` params and a frozen mapper; start from the truth perturbed by `--perturb` (default 0.05; pitch starts at truth unless `--perturb-pitch`) and report parameter and pitch error before/after. **Pass/fail:** exit 1 unless the error of the parameters that affect the audio falls. Or
  - (b) `--target-mode wav` (default): load the target WAV; start from its `.npz` params.
- Synthesis (revision 2): `--mapper formant` (default) or `mlp`; `--source engine` (default: VocalTract pulse at 20 kHz with complex resonators, resampled to 16 kHz), `vocaltract` (band-limited pulse at 16 kHz) or `sawtooth`.
- Trainable objects: `params` (the free variable), the mapper (`--train-mapper/--no-train-mapper`; default on for WAV targets, off for self targets), and for WAV targets a **recording stage** outside the 13 parameters: a learned output gain (`--output-gain`; the WAVs are peak-normalized, so their level says nothing about `ga`) and a learned additive white-noise floor (`--noise-floor`). Both default on for WAV targets, off for self targets.
- `params` are optimized directly and clamped to [0, 1] after each step (projected gradient descent). A sigmoid parameterization cannot reach the many exact zeros in the targets.
- `--freeze name[=value],...` holds parameters fixed, e.g. `--freeze aa=0` removes aspiration noise.
- Optimizer: `Adam(lr=1e-3)`, with pitch (`gp`) at `lr × 0.1` (`--pitch-lr-scale`) and cosine learning-rate decay (`--no-lr-decay` to disable). The best step is kept.
- Loss: `MultiScaleSTFTLoss(log_floor_db=80)` + `0.1 ×` L1 on the waveform (`--log-floor-db`). Noise uses a fixed seed every step.
- Loop for a configurable number of steps (default 500), logging loss every 50 steps.
- Save the best prediction to `out/phase1_overfit.wav`, plus `_target.wav` and `_initial.wav` for listening.
- Print initial and best loss, log-spectral distance (LSD) to the target before/after, median pitch, and **formant error measured from the audio** (`vocaltract.formants`: LPC formant tracks of output vs target, F1/F2/F3). LSD alone does not show whether formants move. Exit 0 if the loss decreased (and, in self mode, parameters moved toward the truth), 1 if not or on divergence, 2 on a bad configuration.

**Why the gain and the loss floor (revision 2):** scanning the loss around the true parameters of `vt_vowel_ah` showed the minimum was *not* at the truth until both were added. Without a free gain the true parameters are 3.6 dB too quiet even at `ga = 1`, so the optimizer made the output louder by widening bandwidths, tilting the spectrum and moving formants. With gain and floor, the loss minimum is at the true formants and bandwidth, and a full fit keeps `f1`–`f4` and `ga` at the engine's values and learns bandwidths of 101–102 Hz (the engine's arithmetic gives 100.6 Hz).

**Expected Output:**
- `python -m vocaltract.phase1.overfit_demo` runs to completion in under 2 minutes on CPU (~55–80 s measured with the 20 kHz engine source).
- `out/phase1_overfit.wav` is written and its STFT visually resembles the target (manual check).
- Loss curve is decreasing on average (windowed trend); median pitch ends within 1% of the target's; on `vt_glide_ah_ee` the output's F1/F2 follow the glide within ~30 Hz on average (`test_glide_tracks_formants`).

**Measured results** (default settings, 500 steps; LSD in dB, lower is better; formant error measured from audio, F1/F2/F3 in Hz):

| Target | LSD start → end | Formant error end | Previous LSD (MLP, before revision 2) |
|---|---|---|---|
| synthetic_vowel_ah | 10.8 → 8.2 | 4 / 9 / 14 | 12.1 |
| vt_vowel_ah | 11.8 → 7.4 | 5 / 27 / 22 | 15.7 |
| vt_glide_ah_ee | 12.0 → 7.8 | 7 / 31 / 29 | 13.6 (F2 error 904 Hz) |
| vt_sung_oo_vibrato | 22.0 → 8.2 | (unreliable: f0 196 Hz + vibrato) | 13.7 |
| vt_seven | 27.1 → 18.1 | (unreliable: consonants) | 11.8 |
| speech_male | 24.2 → 14.3 | (unreliable: unvoiced frames) | 12.7 |
| speech_female | 33.8 → 12.6 | (unreliable: unvoiced frames) | 10.5 |

Pitch ends within 0.5 Hz of the target on every clip. `vt_seven` and the speech clips score worse than with the MLP: they contain frication and nasals that Phase 1 does not model, and the unconstrained MLP filter could imitate them by shaping noise; four formant resonators cannot. That is the expected cost of making the parameters mean what they say.

#### Addendum — review of the Step 7 output (2026-10-11; full notes and reproduction in `docs/reviews/phase1_step7_review.md`)

*Status: addressed by revision 2 (formant mapper, engine source); kept for the record. Current state: `docs/reviews/phase1_completion_review.md`.*

**Measured** (`vt_vowel_ah`, `--freeze aa=0`, 500 steps, default order-10 mapper):

| | Target | Result |
|---|---|---|
| Pitch (Praat) | 119.96 Hz | 119.75 Hz (3 cents) |
| F1 / F2 / F3 / F4 | 725 / 1082 / 2440 / ~3400 Hz | 658 / 1080 / 2383 Hz / missing |
| Harmonic-level RMS error (compare pane) | | 10.2 dB below 4 kHz, 32.3 dB above (`scripts/step7_diagnostics.py`: 8.8 / 28.2) |

**Recommended additions to the demo's report** (cheap, and they make the two remaining gaps visible in numbers):
- The trained filter's pole frequencies and bandwidths at the middle frame.
- Harmonic-level RMS error below and above 4 kHz: levels sampled at k·f0, each clip relative to its strongest
  harmonic. `vocaltract.analysis.compare.harmonic_levels` computes the levels.

**Targets once the Step 2 and Step 4 addenda are in:** harmonic RMS error ≤ 5 dB below 4 kHz and ≤ 10 dB above,
with all four formants within 5%.

**To inspect visually:** `python -m vocaltract.analysis.compare --serve out`, then pick `*_target.wav`, the
result and `*_initial.wav` as tracks. The **harmonic levels** pane plots each track's harmonic envelope (voiced
frames in view) and the difference vs track 1, with the RMS errors above.

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
- `pytest -q` passes all tests. `pytest -q -m "not slow"` runs in under 60 seconds on CPU; the two `slow` tests (full overfit runs: the `vt_glide_ah_ee` formant-tracking regression and self-mode parameter recovery) add ~1.5 minutes.

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

- **Resolved in revision 2: formant parameters were not identifiable** through the MLP mapper (self-mode parameter error grew while the loss fell; the glide target held one vowel). `FormantToLPC` recovers perturbed formants from ~140 Hz error to under ~20 Hz in self mode.
- **Resolved in revision 2: harmonic levels were off the targets** (a spare pole pair stuck at 4.7 kHz and a fixed 1/k source tilt; see the Step 2/4 addenda). After the formant mapper and engine source, harmonic-level RMS error on the VocalTract targets is 0.2–0.7 dB below 4 kHz (`docs/reviews/phase1_completion_review.md`).
- **Pitch is only locally learnable.** Harmonic k moves k times as far as f0, so the spectral loss only pulls pitch back from within ~6 cents; a 64-cent offset does not recover at all. Self mode therefore starts pitch at the truth by default (`--perturb-pitch` to test it), and WAV demos start from the `.npz` pitch. Later phases need pitch from an estimator (or a pitch-specific loss), not from the spectral loss.
- **Frication and nasals are not modelled** (`fa`, `ff`, `na`, `nf` are ignored by the source and filter), so `vt_seven` and the speech clips have a loss floor. The original Phase 1 plan defers these to later phases.
- **The engine source is slower:** rendering at 20 kHz and resampling makes a 500-step run ~55–80 s instead of ~20–30 s. `--source vocaltract` renders at 16 kHz with a band-limited pulse.
- **Vibrato is not modelled** (`vp`, `vr`): pitch is fitted per frame instead.

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
