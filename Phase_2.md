# Phase 2: Complete source model and utterance fitting

Plan: `docs/reviews/phase1_completion_review.md` ("Proposed Phase 2 rescope"). Model every VocalTract
parameter differentiably, fit whole recorded sentences starting from the level-A analysis, and measure
progress as Whisper WER on the ten-sentence set in `assets/recordings/`.

**Gate:** corpus WER ≤ 15% with no clip above 40% (stretch: ≤ 8%).

| Step | Status |
|---|---|
| 2.1 Utterance fitting and WER harness | **done** (below) |
| 2.2 Frication source (`fa`, `ff`) | next; level A's frication is a fixed engine-rendered track until then |
| 2.3 Nasal anti-formant (`na`, `nf`) | |
| 2.4 Vibrato (`vp`, `vr`) | |
| 2.5 Voice-quality source for real voices | |

---

## Step 2.1: Utterance fitting and WER harness

**Code:** `src/vocaltract/phase2/utterance.py` (model and fit), `src/vocaltract/phase2/harness.py` (CLI),
`src/vocaltract/analysis/harmonic_error.py` (moved from `scripts/overfit_report.py`), `tests/test_utterance.py`.

```bash
python -m vocaltract.phase2.harness                 # all ten sentences, ~1.5 h on 8 CPU cores
python -m vocaltract.phase2.harness --clips 2961-960-0019 --steps 50 --out out/p2_try
python -m vocaltract.phase2.harness --report-only   # rebuild out/phase2/report.md
```

### How a sentence is fitted

1. Level A (`vocaltract.analysis.resynth`, default settings) analyses the recording into engine frames, one
   every 9.8 ms.
2. The engine reaches frame k at (k + 1) · 9.8 ms, interpolating linearly from all zeros. That trajectory is
   sampled at the pipeline's frame centres, on a 10 ms grid by default (`--hop 320` gives 20 ms).
3. The model is the formant-mapper pipeline with the engine source, plus the parts the pipeline can't make yet:

   prediction = gain · (voice(params) + fric_gain · frication) + noise floor

   The frication track is the reference engine's render of level A's `fa`/`ff` alone, held fixed.
4. Adam runs for 300 steps on the multi-scale log-spectral loss plus 0.1 · L1. It updates the frames, the mapper's
   bandwidths and tilt, the output gain, the noise floor and the frication gain.

### Design decisions, each from a measurement

- **The whole sentence in one pass, not overlapping windows.**
  - A 15 s sentence at a 10 ms hop takes about 1.5 GB and 1.5 s per step.
  - The windows in the plan were there to bound memory, which isn't needed at this length.
  - One pass has no phase resets or cross-fades.
  - Revisit for long recordings.
- **No per-frame gain normalization** (`normalize_gain=False`).
  - With the formant mapper, the unnormalized pipeline follows the engine's level frame by frame. On a level-A
    sentence the per-frame level difference from the engine was constant to within 0.7 dB (interquartile range). With normalization,
    the interquartile range was 6 dB.
  - Level A calibrates `ga`/`aa` against the engine's level, so this keeps its amplitudes meaningful.
- **Aspiration at the engine's level** (`source.ENGINE_NOISE_RMS`). This fixes a bug in the Phase 1 engine
  source.
  - In the engine, aspiration is uniform noise at a quarter of the glottal pulse's peak: RMS 0.144, not 1.
  - The pipeline's unit-variance noise was 17 dB too loud relative to voicing. After the fix the two agree
    within 0.5 dB.
  - Phase 1's VocalTract targets have `aa` = 0, which is why this went unnoticed.
- **Frication as a fixed engine track.**
  - The engine is linear in its sources up to 32-bit wraparound, and level A backs its gain off until the
    render doesn't overflow. So the voice and frication renders add up exactly to level A's render.
  - The track is converted to pipeline units with the engine/pipeline level ratio measured on the voice part.
  - Step 2.2 replaces it with a differentiable frication source.
- **Whisper beam search (5) and several noise seeds.** Greedy WER is unstable on synthetic speech.
  - Three aspiration-noise realizations of the same initial frames scored 86%, 28% and 24%. The 86% was 25 of
    29 words dropped.
  - The renders were near-identical by every other measure: band levels within 2 dB, the same HNR and voicing,
    and pitch within 10 cents.
  - The harness therefore decodes with beam 5. Each pipeline condition is scored with 3 noise seeds, which
    differ from the training seed, and the WER is their mean.
  - Level A (the engine's LFSR is deterministic) and the original are scored once.

### Results (2026-10-12, `out/phase2/report.md`)

Whisper small.en, beam 5. Settings: 10 ms hop, 300 steps, lr 1e-3. About 7 minutes of fitting per sentence.

| Clip | Sex | f0 | Original | Level A | Init | **Fit** | Harmonic dB < / > 4 kHz: level A → fit |
|---|---|---|---|---|---|---|---|
| 198-209-0000 | F | 212 Hz | 2% | 43% | 45% | **38%** | 9.1 / 5.6 → 6.3 / 5.7 |
| 5703-47212-0000 | M | 83 Hz | 8% | 58% | 52% | **20%** | 10.6 / 5.9 → 7.1 / 3.5 |
| 2094-142345-0006 | F | 194 Hz | 0% | 8% | 12% | **3%** | 10.0 / 13.0 → 4.6 / 13.4 |
| 2961-960-0019 | F | 167 Hz | 0% | 28% | 23% | **20%** | 5.1 / 5.1 → 2.4 / 5.0 |
| 3575-170457-0015 | F | 199 Hz | 0% | 16% | 12% | **0%** | 7.5 / 5.6 → 4.4 / 5.5 |
| 8463-294828-0035 | F | 145 Hz | 0% | 31% | 27% | **2%** | 3.3 / 15.1 → 2.9 / 15.4 |
| 7729-102255-0015 | M | 91 Hz | 0% | 14% | 11% | **8%** | 7.9 / 10.6 → 4.7 / 11.8 |
| 8230-279154-0013 | M | 105 Hz | 0% | 5% | 3% | **1%** | 9.0 / 7.5 → 4.9 / 7.4 |
| 7127-75946-0002 | M | 125 Hz | 0% | 25% | 23% | **6%** | 4.8 / 5.2 → 3.6 / 5.6 |
| 6930-75918-0001 | M | 156 Hz | 0% | 109% | 178% | **43%** | 4.6 / 14.0 → 2.6 / 9.0 |

| Condition | Corpus WER | Median clip | Worst clip |
|---|---|---|---|
| Original | 1.0% | 0% | 8% |
| Level A (engine) | 34.5% | 26% | 109% (6930) |
| Init (pipeline, before fitting) | 40.1% | 23% | 178% (6930) |
| **Fit** | **14.2%** | **7%** | **43% (6930)** |

**Acceptance (2.1): pass.**
- Every clip improves on its level A WER, and the corpus WER falls from 34.5% to 14.2%.
- The fit already meets the Phase 2 corpus gate (≤ 15%). It misses the per-clip gate only on 6930 (43% against
  ≤ 40%).

Notes:
- **Beam 5 vs the manifest.** The manifest's level-A figures (36% corpus) were greedy. With beam 5, level A
  scores 34.5%, and the original 5703 rises from 3% to 8%.
- **Spread across seeds.** The fit's WER varies little with the noise seed (e.g. 7127: 9/5/5%), but the initial
  render's varies a lot (6930: 263/107/163%). Fitting makes the result robust as well as more intelligible.
- **Where the gain is.** It's mostly below 4 kHz: harmonic error falls by 2–5 dB there in most clips. Above
  4 kHz, error stays at 5–15 dB, because that band is mostly frication, which the fit can only rescale
  (fric_gain ended between 0.45 and 1.9).
- **Learned bandwidths** stayed close to the engine's 100 Hz (90–108 Hz), as they should for a VocalTract voice.
- **6930** (fast, breathy, hiss floor) is still the worst clip. It is the natural test for 2.2 (frication) and
  2.5 (voice quality).
