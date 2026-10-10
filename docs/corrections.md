# Phase 1 Corrections

This documents bugs found and fixed while reconciling `src/vocaltract/params.py`
and `src/vocaltract/mapper.py` against `spin/VocalTract.spin` (the original
hardware source) and getting the Phase 1 Step 1/2 test suite (`test_params.py`,
`test_mapper.py`) green under a real package install.

Cross-reference: `/Vault/business/VocalTract_TTS/Gear Emulator Reference.md`
flagged exactly this risk ahead of time — "the core risk of Phase 1 is the
semantic ambiguity of parameter mapping" — and recommended extracting the
parameter table from the original source as "the lowest-cost, highest-value
action." This doc is that reconciliation.

## 1. Parameter scaling bug (the main finding)

`params.py` denormalized frequency/rate parameters (`f1-f4`, `nf`, `ff`, `vr`)
with:

```python
result[name] = val * FORMANT_SCALE * 100   # val is normalized [0,1]
```

But every VocalTract parameter is a single **byte** (0..255) on the hardware —
confirmed by `spin/VocalTract.spin`'s own header table and by its `VAR byte
aa,ga,gp,vp,vr,f1,f2,f3,f4,na,nf,fa,ff` declaration. The amplitude parameters
(`aa`, `ga`, `na`, `fa`) in the same file correctly used `val * 255.0`, but the
frequency parameters used `val * 100` instead — an internally inconsistent,
invented placeholder with no basis in the byte range. The error is a constant
factor of `255 / 100 = 2.55x` on every frequency/rate parameter.

Verified against the hardware doc table itself:
- `f1`: raw 40 → 781 Hz. `40 * 19.53 = 781.2` ✓ (raw, not `raw/2.55`)
- `ff`: raw 60 → 2344 Hz ("Sh"). `60 * 39.06 = 2343.6` ✓
- `f4`: raw 179 → 3496 Hz. `179 * 19.53 = 3495.87` ✓

**Fix:** introduced `BYTE_SCALE = 255.0` and uniformly compute
`raw = normalized * BYTE_SCALE` for all 13 parameters before applying any
physical-unit scale factor.

## 2. `gp`/`vp` used the wrong *kind* of formula, not just the wrong scale

The original code treated glottal pitch (`gp`) as a **linear offset** from a
base value:

```python
offset = (val - 0.5) * PITCH_BASE
result["gp"] = PITCH_BASE + offset
```

But `VocalTract.spin`'s ASM proves `gp` is a **log-frequency value in
1/48-octave units**: `gp` and `vp` are summed together (`add t1,gp`) and then
passed through an `antilog` routine (exponentiation) before becoming a phase
increment — see lines ~319–350 of `spin/VocalTract.spin`. This is also exactly
what the hardware doc's unit column says: `gp` unit is "1/48 octave," with
"100 → 110.00Hz (A2)."

**Fix:**
```python
Hz = GP_BASE_HZ * 2 ** ((raw - GP_BASE_RAW) / OCTAVE_UNIT)   # GP_BASE_HZ=110, GP_BASE_RAW=100, OCTAVE_UNIT=48
```
This reproduces the doc's one fixed point exactly (raw=100 → 110 Hz) and is
monotonic increasing in raw value, matching the old test's directional checks.

`vp` shares `gp`'s log-pitch domain — it's a **modulation depth in octaves**,
not an independent frequency. The doc gives exactly one calibration point:
raw=48 → ±1/2 octave swing. We fit `swing_octaves = raw / VP_SWING_DIVISOR`
with `VP_SWING_DIVISOR = 96` (`48/96 = 0.5`, matching the doc). This is a
best-effort calibration from a single data point, not a bit-exact derivation
of the hardware's fixed-point sine-modulation math — flagged here per the
vault's own guidance ("don't expect to port the source bit-for-bit... use the
source to calibrate semantics"). It should be validated against GEAR emulator
output in Phase 2/3 once real audio comparison is in scope.

`denormalize_batch()` was a byte-for-byte duplicate of `denormalize()` (same
bug, twice); it now just calls `denormalize()`.

## 3. `mapper.py`: shape contract didn't match its own docstring

`ParamToLPC.forward()` claimed to support `(B,T,13)`, `(T,13)`, and `(13,)`
inputs, but the old implementation used `batch, time, _ = params.shape`
unconditionally, which raises on anything but a 3D input. Fixed to promote
lower-rank inputs to `(1,T,13)` / `(1,1,13)` first, so the output is always
`(B,T,order)`.

## 4. `mapper.py`: `_levinson_durbin` broke on non-2D input

`a_prev = a[:, :m-1]` hardcoded a slice on dimension 1, which only happened to
work because `forward()` always flattens to a 2D `(B*T, order)` tensor before
calling it — contradicting the function's own docstring, which claims generic
support for "(*, order) where * can be batch, time dimensions." Fixed to
`a[..., :m-1]`.

## 5. `mapper.py`: `to_reflection` was mathematically wrong and mutated its input

Three separate bugs in the Schur/step-down inverse:
- `lpc_coeffs[..., :m+1] = new_a[:m]` sliced the **batch** dimension (`[:m]`)
  instead of the **order** dimension (`[..., :m]`) on the right-hand side,
  and wrote into a slice one element too wide on the left-hand side.
- `k[..., m-1] = k_m` recorded the *current* order's reflection coefficient
  into the *next* order's output slot — a different, incorrect value.
- It mutated `lpc_coeffs` (the caller's input tensor) in place; `.clone()`
  was only ever applied to build `k`, not the working copy.

Rewrote the recursion from the defining pair of Levinson-Durbin equations
(see docstring in `mapper.py`) and verified it inverts `_levinson_durbin` to
float32 precision (max abs error ≈ 3e-4 over a random 2×3×10 batch).

## 6. Test bugs fixed alongside (`test_mapper.py`)

- `test_pole_stability` called `.item()` on plain Python floats (from
  `.tolist()`) and then called `torch.polyroots`, which doesn't exist in
  torch 2.1 (`AttributeError`, not the `RuntimeError` the test was built to
  catch). Replaced with `numpy.roots`, mirroring the already-correct
  stability check in `mapper.py`'s own `__main__` demo.
- `test_inverse_consistency` compared a 10-dim LPC output against the
  original 13-dim `params` tensor — a shape mismatch, and not what
  `to_reflection` claims to invert. Rewrote it to test the actual claimed
  property: reflection coefficients → `_levinson_durbin` → `to_reflection` →
  the same reflection coefficients.

## 7. Package wasn't installable

There was no `pyproject.toml`/`setup.py`, so `pytest` failed at collection
with `ModuleNotFoundError: No module named 'vocaltract'` — Phase 1's own
"definition of done" (`pytest -q` passes) was unreachable without manually
setting `PYTHONPATH`. Added a minimal `pyproject.toml` (setuptools backend,
`src/` layout) and installed with `pip install -e .`.

## Status after these fixes

```
$ pytest -q
24 passed
```

All of Step 1 (`test_params.py`) and Step 2 (`test_mapper.py`) pass under a
real editable install, no `PYTHONPATH` tricks required.

## Known follow-ups not addressed here (out of scope for this pass)

- `mapper.py`'s `__main__` demo block calls `.cpu().numpy()` on a tensor that
  still has `requires_grad=True` in its pole-stability section; needs
  `.detach()`. Harmless (it's a demo script, not covered by any test) but
  worth a one-line fix.
- `vp`'s swing-depth formula (`VP_SWING_DIVISOR`) is a single-point
  calibration, not a verified derivation — see §2. Validate against GEAR
  emulator audio once Phase 2/3 bring in real glottal modeling, per the
  vault's "Gear Emulator Reference" doc.
- Steps 3–7 of `Phase_1.md` (`filter.py`, `source.py`, `losses.py`,
  `phase1/smoke_test.py`, `phase1/overfit_demo.py`) are still unwritten;
  this pass only covers Steps 1–2 plus packaging.
