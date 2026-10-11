# Phase 1 completion review, and a proposed Phase 2 rescope

*Reviewer notes, 2026-10-11 (branch `claude/analysis`), on main at `4260be8` (Phase 1 Step 7 revision 2: formant
mapper, engine source, recording stage). Follows `docs/reviews/phase1_step7_review.md`, whose recommendations
revision 2 addressed.*

## Verdict

**Phase 1 is complete in substance; close the housekeeping items below, then start Phase 2.** The pipeline is now
a differentiable model of the VocalTract voice path: one resonator per formant at `f1`–`f4`, the engine's own
glottal pulse, and rendering at 20 kHz. It reproduces engine renders to a fraction of a dB. It recovers perturbed
formants (the interpretability goal), and on real speech it turns an unintelligible excerpt into a correctly
transcribed one.

One caution when reading the demos: **WAV-mode fits start from the target's `.npz` parameters**, so on VocalTract
targets they start almost on the answer. The near-perfect `vt_glide_ah_ee` overlay is mostly a correctness check
of the engine model, not evidence of learning. The learning evidence is self mode and speech (below).

## Measurements

Outputs: the developer's `out/fm/final_*` (500 steps, default settings). The numbers are reproducible with
`python scripts/overfit_report.py <out/fm> --prefix final_ --asr speech`.

Harmonic-level RMS error is the compare tool's measure. Levels are sampled at k·f0 along the target's Praat pitch
track and averaged over voiced frames. Each envelope is relative to its own peak.

| Clip | Harmonic RMS, initial (< 4 kHz / > 4 kHz) | Final | Notes |
|---|---|---|---|
| vt_vowel_ah | 0.1 / 1.2 dB | 0.5 / 1.1 dB | starts at the answer |
| vt_glide_ah_ee | 1.5 / 2.2 | 0.2 / 1.8 | follows the /a/→/i/ glide |
| vt_sung_oo_vibrato | 1.3 / 3.0 | 0.7 / 1.6 | vibrato fitted per frame (`vp`/`vr` not modelled) |
| synthetic_vowel_ah | 2.9 / 7.6 | 1.5 / 6.9 | not a VocalTract voice; high band limited by the engine pulse |
| vt_seven | 9.3 / 24.3 | 1.1 / **34.3** | frication not modelled; high band gets *worse* |
| speech_male | 22.2 / 10.8 | 5.3 / 5.8 | init = level-A analysis (approximate) |
| speech_female | 13.9 / 20.4 | 3.9 / 4.4 | init = level-A analysis |

Whisper (small.en) on the 1 s speech excerpts:

| Clip | Target | Initial | Final |
|---|---|---|---|
| speech_female | "in the next morning" | "the left floor" | **"in the next morning"** |
| speech_male | "during the summer months" | "during the summer" | "during the summer" (still drops "months") |

Self mode, from the developer's `out/fm/final.txt`. Formants are perturbed by about 135 Hz; pitch starts at the
truth:

| Target | f1–f4 parameter error, before → after |
|---|---|
| vt_vowel_ah | 140/134/137/126 → 10/49/16/6 Hz |
| vt_glide_ah_ee | 140/134/137/126 → 10/16/30/8 Hz |
| vt_seven | 140/134/137/126 → 29/72/44/41 Hz |

That's the interpretability result: formant parameters are now identifiable from audio. (Under the MLP mapper,
parameter error grew.)

The fast test suite passes on main: 262 passed, 1 skipped, 2 slow tests deselected. It was run read-only in the
developer's checkout.

## Close before calling Phase 1 done

- [ ] Tick the Definition of Done boxes in `Phase_1.md` and the Phase 1 box in the README roadmap.
- [ ] Update the README's Phase 1 section to describe the current pipeline. Step 9 of the spec still says
      "Phase 2 will replace the sine excitation", which is out of date.
- [ ] Install `requirements.txt` in a fresh venv and run the suite.
- [ ] Run the 2 slow tests (`pytest -m slow`).

## Known limitations carried into Phase 2

1. **Every fit starts from given parameters.** Pitch is only recovered from within about 6 cents. Fitting from
   nothing needs a pitch estimator for initialization (Praat now; Phase 3's encoder later).
2. **Frication, nasals and vibrato aren't modelled** (`fa`, `ff`, `na`, `nf`, `vp`, `vr`). `vt_seven`'s high band
   gets worse during the fit, because the optimizer bends the voiced spectrum to make up for missing frication.
3. **Real voices aren't shaped like VocalTract's pulse.** The speech fits plateau at 4–6 dB, and the synthetic
   (non-VocalTract) vowel keeps 7 dB of high-band error.
4. **Demos are 1 s clips at a 20 ms frame hop.** Nothing yet fits a whole utterance, and 20 ms is coarse for stop
   bursts and fast consonant transitions.

## Proposed Phase 2 rescope

The roadmap's Phase 2 is "Glottal source model (differentiable LF / wavetable)". Revision 2 already delivered the
core of that: a differentiable VocalTract glottal pulse. Meanwhile the biggest intelligibility losses are elsewhere.
In the level-A resynthesis of the two LibriSpeech sentences (WER 43% female / 47% male, against 2–3% for the
originals), the errors are concentrated in **consonants**: "white paint" → "white tooth", missing and quiet
fricatives and stops in the compare tool's difference pane. So I propose re-centring Phase 2 on **the rest of the
VocalTract source model plus whole-utterance fitting, measured by WER**:

> **Phase 2: Complete source model and utterance fitting.** Model every VocalTract parameter differentiably, fit
> whole recorded sentences starting from the level-A analysis, and measure progress as Whisper WER.

### Steps, in order

**2.1 Utterance fitting and WER harness (first, so every later step is measured).**
- Fit a whole sentence (about 15 s): optimize in overlapping windows (e.g. 1 s windows, 0.25 s overlap, cross-faded
  on output) so memory stays bounded. Initialize from `vocaltract.analysis.resynth`, which already produces
  per-frame VocalTract parameters for any recording (`<name>_frames.csv`, 9.8 ms hop), resampled onto the
  pipeline's frame grid.
- Support a 10 ms frame hop (160 samples at 16 kHz) next to 20 ms.
- Report WER (`vocaltract.analysis.intelligibility`) for: the original, the level-A init, and the fit. Also report
  harmonic error and write a compare page (`vocaltract.analysis.compare`).
- Data: the ten-sentence set in `assets/recordings/` (5 female, 5 male, f0 83–212 Hz; added 2026-10-11, see the
  addendum below). Loop over `manifest.json`. Report **corpus WER** (total errors / total words), the median clip,
  and the worst clip.
- *Acceptance:* the harness runs end to end on all ten. The fit's corpus WER is no worse than level A's (36%), and
  no clip is worse than its own `level_a_wer`.

**2.2 Frication source (`fa`, `ff`).**
- Engine behaviour (`src/vocaltract/reference/engine.py`, "frication" block): a sine at `ff` whose phase is jittered
  by the LFSR noise every sample, scaled by `fa`. It is added **after** the formant cascade and the nasal stage, so
  formants don't shape it (unlike aspiration `aa`, which enters before the cascade).
- Differentiable options:
  - (a) Exact: phase = cumsum(ff increment + scaled noise), as the pitch phase does. This is faithful, but `ff` is
    then only locally learnable, like pitch.
  - (b) Equivalent spectrum: white noise through a resonator at `ff`, with the bandwidth set to match the jitter
    spread, scaled by `fa`. This gives smooth gradients in `ff`.
  - Recommend (b) for fitting, validated against (a) and the engine on `vt_seven`.
- *Acceptance:* the `vt_seven` high band (> 4 kHz, all frames, not only voiced) within 6 dB. Self mode recovers
  `fa`/`ff` on `vt_seven`. WER on the sentences improves on step 2.1.

**2.3 Nasal anti-formant (`na`, `nf`).**
- Engine: after the formant cascade, a resonator at `nf` whose output, scaled by `na`, is subtracted on the next
  sample. That's a zero-pair (anti-resonance) stage. Revision 2's FIR-numerator support (`numerator()`,
  `filter.fir_frames`) is the natural place for it.
- *Acceptance:* self mode recovers `nf` on `vt_seven`'s "n". The "n" segment's spectrum within 3 dB.

**2.4 Vibrato (`vp`, `vr`).**
- Engine: a sine LFO at rate `vr` adds `vp`/96 octaves of swing to `gp` before the antilog. Cheap to add, and it
  replaces per-frame pitch fitting on sung targets.
- *Acceptance:* `vt_sung_oo_vibrato` fitted with a constant `gp` and learned `vp`/`vr` within 5% of the truth.
  Lowest priority for speech, so it can slip to a later phase.

**2.5 Voice-quality source for real voices.**
- Keep the VocalTract pulse as the default and special case. Add per-utterance (or slowly varying) voice-quality
  parameters:
  - **pulse shape:** e.g. an LF-model Rd parameter, or a learnable exponent on the engine's antilog pulse
  - **spectral tilt**
  - **breathiness:** the aspiration mix within voiced frames
- Level A showed that breathiness is what made the female voice intelligible at all (WER 53% → 43%).
- These are global or low-rate controls, not new frame parameters. The 13-parameter VocalTract schema stays the
  interface, so back-porting to the engine is still possible.
- *Acceptance:* speech harmonic error below 3 dB under 4 kHz on the excerpts. The synthetic vowel's high-band error
  below 3 dB. WER improves on 2.2–2.3.

### Phase 2 goals and non-goals

- **Gate:** corpus WER on the ten-sentence set **≤ 15%** with **no clip above 40%** (level A: 36% corpus, 23.5%
  median clip, worst 133%; originals: 0–3%). Stretch target: ≤ 8% corpus. Listening checks alongside, since Whisper's language model is forgiving on predictable sentences.
- **Interpretability check:** self mode recovers every modelled parameter, now including `fa`, `ff`, `na`, `nf`,
  `vp` and `vr`.
- **Non-goals:**
  - an audio→parameter encoder (Phase 3; Phase 2 starts from the level-A analysis)
  - text input
  - real-time or embedded work
  - GPU-specific optimization

### Effect on the later roadmap

- **Phase 3** (encoder) gets its training targets from Phase 2's utterance fits, and replaces the level-A
  initialization.
- **Phase 4's** "formant tracking loss" partly exists already as `vocaltract.formants` and the harmonic measures.
  Revisit its scope when Phase 3 starts.

## Addendum: the ten-sentence evaluation set (2026-10-11)

`assets/recordings/` now has ten sentences: the original two, plus eight from LibriSpeech test-clean (4 female,
4 male, chosen so the originals score 0% WER). `manifest.json` records each clip's original and level-A WER. Level A
(`python -m vocaltract.analysis.resynth <clip> --asr small.en`, default settings):

| Clip | Speaker | f0 | Original WER | Level-A WER |
|---|---|---|---|---|
| 8230-279154-0013 | M | 105 Hz | 0% | 3% |
| 2094-142345-0006 | F | 194 Hz | 0% | 8% |
| 2961-960-0019 | F | 167 Hz | 0% | 17% |
| 7729-102255-0015 | M | 91 Hz | 0% | 20% |
| 8463-294828-0035 | F | 145 Hz | 0% | 22% |
| 7127-75946-0002 | M | 125 Hz | 0% | 25% |
| 3575-170457-0015 | F | 199 Hz | 0% | 29% |
| 198-209-0000 | F | 212 Hz | 2% | 43% |
| 5703-47212-0000 | M | 83 Hz | 3% | 47% |
| 6930-75918-0001 | M | 156 Hz | 0% | **133%** |
| **Corpus** | | | | **36%** (144 errors / 400 words) |

**What this changes:**
- The two original clips were the hardest of the set, not typical. The "43–47%" baseline overstated the gap, so
  the Phase 2 gate above has been restated on the corpus.
- **6930 is a hard case worth keeping.** A fast, breathy reader on a recording with an audible hiss floor. Pitch
  tracking is fine (median error 9 cents). But level A voices only 16% of frames (the original: 42%) and renders
  much of the speech as frication and aspiration noise, so Whisper hallucinates. It is a direct test of Phase 2's
  frication (2.2) and voice-quality (2.5) work, and of noise robustness in the analysis.
- Spread matters: per-clip WER ranges from 3% to 133% on the same settings, so report the corpus figure, the median
  and the worst clip, not one sentence.
