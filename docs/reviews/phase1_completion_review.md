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
- Data: the two LibriSpeech sentences in `assets/recordings/`. Add a few more (5–10, mixed speakers) so WER isn't
  decided by two sentences. I can add these.
- *Acceptance:* the harness runs end to end on both sentences, and the fit's WER is no worse than level A's
  (43% / 47%).

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

- **Gate:** WER on the sentence set at **≤ 20%** (level A: 43–47%; originals: 2–3%), with a stretch target of
  ≤ 10%. Listening checks alongside, since Whisper's language model is forgiving on predictable sentences.
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
