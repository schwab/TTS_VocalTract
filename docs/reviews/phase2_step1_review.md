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

Reproduce with `python scripts/engine_export_check.py out/phase2 assets/recordings out/phase2_engine`. Export:
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
