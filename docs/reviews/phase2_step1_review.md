# Phase 2 step 2.1 review: the fit is measured on the pipeline, not on the engine

*Reviewer notes, 2026-10-12 (branch `claude/analysis`), on `phase2-utterance-fit` at `ba3cf83`. The harness outputs
used are the developer's `out/phase2/`.*

## Summary

2.1 is good work, and the pipeline numbers check out:
- the harness is clean, and corpus/median/worst reporting is in place
- the aspiration-level bug is a real find
- the move to beam search with held-out noise seeds is well justified

**But the harness scores the differentiable pipeline's render, and the fitted frames have never been played by
VocalTract.** The project's claim, and the Propeller phrase-library idea, rest on the 13 bytes. So I exported the
fits to engine frames and scored the reference engine's render:

| Condition | Corpus WER | Clips worse than level A |
|---|---|---|
| Level A (engine) | 34.5% | — |
| Init, exported to the engine | 39.5% | — |
| Fit, pipeline render (the reported number) | **14.2%** | 0 |
| **Fit, exported to the engine** | **26.2%** | **3** (198: 43→53%, 2094: 8→16%, 2961: 28→34%) |

The fit still helps on the engine: corpus 34.5% → 26.2%, and 7 of 10 clips improve, some a lot (8463: 31→4%,
7127: 25→7%, 6930: 109→81%). But half the gain disappears on export, and by the engine's measure step 2.1's own
acceptance ("no clip worse than its level A") fails.

Reproduce with `python scripts/engine_export_check.py out/phase2 assets/recordings out/phase2_engine`. It reads the harness results in `out/phase2`, so run `python -m vocaltract.phase2.harness` first, or pass another checkout's `out/phase2`. Export:
- each frame is moved to the engine's frame instants, (k + 1) · 9.8 ms, with the same linear interpolation as
  `engine_to_grid` but in reverse
- `fa` is multiplied by the learned frication gain (`fa` never clipped)
- values are rounded to bytes and rendered with `vocaltract.reference`
- one Whisper small.en beam-5 decode per clip; the engine is deterministic

## Per clip

| Clip | Level A | Init → engine | Fit (pipeline) | **Fit → engine** |
|---|---|---|---|---|
| 198-209-0000 | 43% | 43% | 38% | **53%** |
| 5703-47212-0000 | 58% | 50% | 20% | **53%** |
| 2094-142345-0006 | 8% | 11% | 3% | **16%** |
| 2961-960-0019 | 28% | 28% | 20% | **34%** |
| 3575-170457-0015 | 16% | 16% | 0% | **2%** |
| 8463-294828-0035 | 31% | 29% | 2% | **4%** |
| 7729-102255-0015 | 14% | 14% | 8% | **11%** |
| 8230-279154-0013 | 5% | 3% | 1% | **0%** |
| 7127-75946-0002 | 25% | 36% | 6% | **7%** |
| 6930-75918-0001 | 109% | 154% | 43% | **81%** |

## Where the gap comes from

> **Update (same day):** finding 2 below doesn't hold up under separate tests. See "Addendum: what the export
> loses" at the end, which replaces this section's conclusions.

I tested three candidates on the clips with the largest gaps (pipeline WER is the mean of 3 seeds):

| Clip | Pipeline fit | Same frames, **untrained mapper** | **Exported bytes, back through the pipeline** | Exported bytes, engine |
|---|---|---|---|---|
| 5703-47212-0000 | 20% | 20% | 35% | 53% |
| 2094-142345-0006 | 3% | 2% | 4.5% | 16% |
| 198-209-0000 | 38% | 38% | — | 53% |

1. **Not the learned mapper.** Rendering the fitted frames with the untrained mapper (no tilt poles, 100 Hz
   bandwidths) gives the same WER. Tilt and bandwidths, the parts the engine can't reproduce, aren't what made the
   fit intelligible.
2. **About half is the export round trip.** Moving the frames from the 10 ms pipeline grid to the 9.8 ms engine
   knots and rounding to bytes, then rendering *through the pipeline*, takes 5703 from 20% to 35%. The mean change
   is about 2 bytes on f1–f4 and 1–1.5 on `aa`/`ga`. The fit uses frame-level detail tied to its own grid, and it
   doesn't survive resampling. The same loss shows on the init conditions: level A 34.5%, the same frames after the
   round trip 39.5%.
3. **The rest is the pipeline vs the engine on identical bytes** (5703: 35% → 53%). It's not yet located.
   Candidates:
   - CORDIC resonators vs the IIR during fast transitions
   - `gp` quantization against the fit's fine pitch moves
   - the LFSR noise vs Gaussian aspiration
   - the engine pulse's aliasing at 20 kHz

## Recommendations

In priority order. 1–3 are cheap and, I think, belong in 2.2's first commit, so 2.2 is measured on the engine
from the start.

1. **Make the engine render a harness condition, and gate Phase 2 on it.** Add `engine`, which plays the fitted
   frames as bytes on `vocaltract.reference` (the export above). Report it next to `fit`, and use it for the
   acceptance checks and the Phase 2 gate. Keep the pipeline number as a diagnostic. Then the WER we report is the
   WER of the 13 bytes.
2. **Fit on the engine's frame grid, so export is exact.**
   - Use one parameter frame per engine frame: 9.8 ms, which is 196 samples at the pipeline's 20 kHz render rate
     (integer, unlike at 16 kHz).
   - Upsample with the engine's own convention: frame k is reached at (k + 1) · 9.8 ms, with a linear ramp from
     frame k − 1. That is `engine_to_grid`'s convention, not `upsample_frames`'s centred one.
   - Export then just rounds. This removes the round-trip half of the gap.
3. **Make the fit aware of byte quantization.** For the last ~50 steps (or throughout), pass the frames through
   straight-through rounding to multiples of 1/255. Then the optimizer can't rely on sub-byte detail the engine
   will never see.
4. **Then find the remaining engine/pipeline mismatch on identical bytes.**
   - Render the same byte frames through both. Report per-frame spectral differences, split by voiced, fricative
     and transition frames, to find where they part.
   - If no single cause appears, add an engine-in-the-loop check every N steps (export, render, measure).
     It isn't differentiable, but it shows when the optimizer is exploiting the model.
   - Phase 1's engine tests were on held vowels and slow glides. Real speech moves formants much faster, which is
     where a CORDIC-vs-IIR difference would show.
5. **Freeze the mapper for speech fits** (`train_mapper=False`), or at least drop the tilt poles. They cost
   nothing in WER (finding 1), and keeping the model to what the engine can do keeps the fit interpretable.
6. **Minor:**
   - `best_step` is 291–300 on every clip, so the fit was still improving when the cosine schedule ended. Try 600
     steps on one or two clips to see whether there's more.
   - The manifest's `level_a_wer` is greedy and the harness's level A is beam 5. Add `level_a_wer_beam5`, or a note
     in the manifest README.

## For step 2.2 (frication)

- Option (b) from the rescope (a noise resonator at `ff`) needs its bandwidth **fixed** to the engine's
  jitter-equivalent spread, not learned. Only `fa` and `ff` exist in the engine, so a learned bandwidth has nowhere
  to go on export.
- Judge 2.2 with the engine condition from recommendation 1. A differentiable frication source that the engine
  doesn't reproduce would show the same pattern as above: better pipeline WER, little gain on the engine.
- Validate (b) against the engine on `vt_seven` and on exported speech fricatives, not only on the pipeline.

## Action items

For the developer agent. Each item says what to change, where, and when it's done. Do P2-1 to P2-3 before reporting
any 2.2 results. P2-4 to P2-6 can follow.

- [ ] **P2-1: engine condition in the harness.**
  - *Change:* in `src/vocaltract/phase2/harness.py`, add an `engine` condition to `CONDITIONS`. It exports
    `result.params` to uint8 engine frames (frication gain folded into `fa`), renders them with
    `vocaltract.reference`, resamples to 16 kHz and scores once (deterministic).
  - Put the export in the library, e.g. `vocaltract.phase2.utterance.to_engine_frames`. It can start from
    `to_engine_frames` in `scripts/engine_export_check.py`.
  - Save the frames as `<id>_engine_frames.csv` (or `.npz`) and write `<id>_engine.wav`. Add the engine render as
    a track on the compare page.
  - In `report.md`, show `engine` in both tables. Change the 2.1 acceptance check and the Phase 2 gate in
    `Phase_2.md` to use `engine` WER. Keep `fit` (pipeline) as a diagnostic.
  - *Test:* a fitted-params → engine-frames round trip on a tiny synthetic clip gives uint8 frames of the expected
    length, and `fa` is scaled by the gain.
  - *Done when:* `python -m vocaltract.phase2.harness --report-only` (after refits) shows engine WER for every clip.
    On the current fits it should reproduce about 26% corpus.

- [ ] **P2-2: fit on the engine's frame grid.** *(Still worth doing so export is exact, but the addendum shows the
  grid change alone isn't the main loss. Do P2-8 first.)*
  - *Change:* add a frame mode to `utterance.fit` in which each parameter frame is one engine frame (9.8 ms; 196
    samples at the 20 kHz render rate). Upsample with the engine's convention: frame k is reached at (k + 1) · 9.8
    ms, with a linear ramp from frame k − 1 and from zeros before frame 0, i.e. the `engine_to_grid` trajectory.
  - Level A's frames then initialize the fit directly, with no `engine_to_grid` resampling. Export is a plain
    `round(p · 255)`.
  - *Test:*
    - with the formant mapper frozen, the pipeline's per-sample parameter trajectory equals the engine's
      interpolation for the same frames (within 1e-6)
    - init frames → export reproduces level A's bytes exactly
  - *Done when:* the "init → engine" condition equals level A's WER exactly (same bytes), and the engine-vs-pipeline
    WER gap on the fit is re-measured and reported.

- [ ] **P2-3: quantization-aware fitting.** *(Low priority: rounding alone costs nothing measurable; see the
  addendum.)*
  - *Change:* add a `FitConfig.quantize_steps` setting, default 50. For the last N steps, `FrameParams.assemble()`
    returns `p + (round(p · 255) / 255 − p).detach()` (straight-through). The returned best params are the rounded
    ones.
  - *Done when:* exported bytes equal `round(best_params · 255)` without further change. Engine WER is reported with
    and without quantization on 2 clips (5703, 2094).

- [ ] **P2-4: locate the remaining engine/pipeline mismatch.**
  - *Change:* write a diagnostic (script or harness flag) that renders the same byte frames through the engine and
    through the pipeline (frozen mapper, frication track). It reports the per-frame log-spectral difference, split
    into voiced, fricative/aspirated and fast-transition frames (|Δf1| + |Δf2| above a threshold).
  - It also renders ablations: Gaussian vs LFSR-like uniform noise, and pitch rounded vs not.
  - *Done when:* there's a short written finding naming the dominant source of the gap, with numbers. Then either
    fix the pipeline to match, or add an engine check every N steps that logs engine WER or engine spectral loss.

- [ ] **P2-5: freeze the mapper for speech fits.**
  - *Change:* set `FitConfig.train_mapper=False` (or `tilt_poles=0`) as the default for utterance fits. Note the
    measurement in `Phase_2.md`: untrained mapper = same WER (20/2/38% vs 20/3/38% on 5703/2094/198).
  - *Done when:* the default is changed and the harness reruns with no WER regression on the pipeline.

- [ ] **P2-6: housekeeping.**
  - Run 600 steps on 5703 and 2094 and report whether engine WER keeps improving: `best_step` was 291–300 on all
    clips.
  - Add `level_a_wer_beam5` to `assets/recordings/manifest.json`, or a sentence in its README saying the manifest's
    figures are greedy.

- [ ] **P2-7 (step 2.2 scope): frication must be engine-exportable.**
  - If 2.2 uses a noise resonator at `ff`, fix its bandwidth to the engine's jitter-equivalent spread; don't learn
    it.
  - Accept 2.2 on the **engine** condition: `vt_seven` high band within 6 dB of the engine render of the fitted
    bytes, and engine corpus WER improves on P2-2/P2-3.

## Addendum: what the export loses (2026-10-12, later)

Follow-up experiments on 5703, 2094 and 8463, rendering the same fitted frames in different ways. Pipeline WER is
the mean of 3 noise seeds. Engine WER is one decode.

### What it isn't

Each candidate was tested on its own, through the pipeline:

| Perturbation of the fitted frames | 5703 | 2094 | 8463 |
|---|---|---|---|
| none (the fit) | 20% | 2% | 2% |
| rounded to bytes on their own grid | 19% | 2% | 1% |
| resampled to the 9.8 ms engine instants and back, no rounding | 17% | 4% | 2% |
| untrained mapper (no tilt, 100 Hz bandwidths) | 20% | 2% | — |
| formants **ramped** (coefficients held 1 ms instead of 10 ms, like the engine's ramps) | 26% | 0% | 1% |

None of these explains the gap. The earlier "about half is the export round trip" figure (5703: 35%) was one
combined measurement; tested separately, resampling and rounding are each harmless. With Whisper's spread on a
36-word clip (single-seed values ranged 14–44% here), I no longer trust that number.

### What it is

1. **On smooth frames, engine and pipeline agree.** Level A's frames through the ramped pipeline score 50/10/28%,
   against 50/11/29% on the engine. Spectra on identical bytes agree within 1–3 dB per band below 4 kHz.
2. **The fit's gain sits in frame-to-frame jitter.**
   - The fit makes f1–f4 zig-zag. The mean frame-to-frame change grows from about 2 bytes (level A) to 7–10 bytes,
     i.e. 140–200 Hz every 10 ms.
   - Half of all the change the fit makes is this fast component (it stays after a 50 ms moving average).
   - It's everywhere: largest in silent and fricative frames, where formants are barely observable (5–6 bytes),
     and 3–5 bytes in voiced frames.
   - Removing it loses most of the gain, through the pipeline:

   | | 5703 | 2094 | 8463 |
   |---|---|---|---|
   | fit | 20% | 2% | 2% |
   | fit, 3-frame moving average | 36% | 11% | 9% |
   | init + only the slow part of the fit's change | 37% | 23% | 4% |

3. **The engine doesn't render that jitter the way the pipeline does.** On identical, ramped trajectories, voice
   only:
   - engine 16% vs pipeline 6% on 2094
   - engine 50% vs pipeline 43% on 5703
   
   Yet per-frame levels match (within 0.3 dB in fricative frames) and band spectra match within 1–4 dB, so the
   difference is in fine detail that these measures don't capture. My leading hypothesis is resonator structure
   under fast frequency changes:
   - The engine's CORDIC resonators rotate their state, so a frequency change doesn't change the stored energy.
   - A direct-form IIR filter's state is reinterpreted when its coefficients change, which modulates amplitude
     and leaves transients.
   - So the optimizer may be exploiting direct-form artefacts.
   
   That's not proven. Aspiration noise type and the engine's 700 µs parameter stepping are the other suspects.

**In short:** the optimizer found a fragile solution that relies on details of the differentiable model the
engine doesn't share. The robust part of the improvement is smaller than 14.2% suggests, but it's real: the engine
corpus WER still improves from 34.5% to 26.2%.

### Revised action items (supersede the priorities above; P2-1 stays first)

- [ ] **P2-8: temporal smoothness on the formant tracks (do right after P2-1).**
  - *Change:*
    - Add a penalty on the first or second difference of f1–f4 (and possibly `ga`/`aa`), in bytes per frame, to the
      utterance loss.
    - Where formants are unobservable (`ga`, `aa` ≈ 0), tie them to their neighbours (stronger penalty) instead
      of letting Adam's normalized steps random-walk them.
  - *Done when:* there's a sweep of the penalty weight on 3 clips (5703, 2094, 8463), reporting pipeline WER,
    engine WER (P2-1) and mean frame-to-frame formant change. Pick the weight that minimizes **engine** WER. The
    expectation is that the pipeline–engine gap closes, at some cost in pipeline WER.
  - It also serves interpretability: 150–200 Hz zig-zags every 10 ms aren't articulation.
- [ ] **P2-9: engine-faithful resonators under fast changes (if P2-8 doesn't close the gap).**
  - *Change:* implement the formant sections in the engine's rotation form (a decaying 2-D rotation of the state
    per sample, frequency from `f`), with parameters stepped every 14 samples (700 µs) at 20 kHz as the engine does.
    It's differentiable, and stable under interpolation by construction.
  - *Done when:* engine and pipeline WER on the current (jittery) 2.1 fits agree within a few points.
- [ ] **P2-10: a less noisy engine measurement.**
  - The engine render is deterministic but Whisper isn't robust: a single decode is one draw. Score the engine
    condition over 3 small perturbations (e.g. 0/1/2 ms leading silence, or ±0.1 dB dither) and average, as the
    pipeline already does with noise seeds.
  - Report per-clip spread. On ~40-word clips, ±10 points per clip is normal, so judge on corpus WER.
