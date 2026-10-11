# Phase 1 Step 7 review: the `vt_vowel_ah` overfit result

*Reviewer notes, 2026-10-11 (branch `claude/analysis`). For the developer of Phase 1. The actionable parts are also
in `Phase_1.md` as addenda to Steps 2, 4, 5 and 7. This file has the full reasoning, the measurements, and the
corrections to my earlier advice.*

## Summary

The revised pipeline (gain-normalized stable filter, band-limited harmonic source, trainable mapper) is a big step
forward. On a real VocalTract target it now gets pitch within 3 cents and F1–F3 within 9% / 0% / 2%, with most
harmonics from 0.8 to 3.6 kHz within ±6 dB. Two gaps remain, and both have measured causes:

1. **A spare pole pair sticks at 4.7 kHz.** It makes a resonance the target doesn't have, and +30 to +50 dB too
   much energy above 4 kHz. F4 (3.4 kHz) is never modelled.
2. **The fixed 1/k source tilt is wrong for VocalTract.** The filter compensates by widening F1/F2, which leaves
   the low harmonics too strong and the F2–F3 valley too shallow.

Recommended fixes: a **formant-parameterized filter** (one resonator per formant at `f1`–`f4`) and a **learnable
source tilt**. The first also addresses the "parameters are not identifiable" limitation.

## What was measured

- **Target:** `out/phase1_overfit_aa0_target.wav`. Correlation 1.0 with `assets/phase1/vt_vowel_ah.wav`: a
  real VocalTract render (f0 120 Hz; formants 730 / 1090 / 2440 / 3400 Hz), not a self-generated target.
- **Result:** `out/phase1_overfit_aa0.wav` (`overfit_demo --freeze aa=0`, 500 steps, order-10 mapper).
- **Reproduce everything below:** `python scripts/step7_diagnostics.py --freeze aa=0 [--order N]`. Runs are
  deterministic (seeded); a run takes about 20–30 s on CPU.
- **Visual check:** `python -m vocaltract.analysis.compare --serve out`. Pick target, result and initial as
  tracks; the **harmonic levels** pane shows the envelopes and their difference.

### Accuracy

| | Target | Result |
|---|---|---|
| Pitch (Praat, median) | 119.96 Hz | 119.75 Hz (3 cents) |
| F1 / F2 / F3 / F4 (Praat Burg) | 725 / 1082 / 2440 / ~3400 Hz | 658 / 1080 / 2383 Hz / missing |
| F1 / F2 bandwidth (Praat) | 49 / 61 Hz | 245 / 217 Hz |
| Log-spectral distance (demo metric) | | 35.8 → 15.6 dB |
| Harmonic-level RMS error, below / above 4 kHz | | 8.8 / 28.2 dB (script) · 10.2 / 32.3 dB (compare pane) |

The two harmonic-error figures use different averaging. The script compares harmonic *numbers* using the target's
pitch. The compare pane interpolates each frame's harmonics onto a 20 Hz frequency grid, which also works when the
two clips' pitches differ. Use either consistently; don't mix them.

### Harmonic levels (result − target, dB, each relative to its own strongest harmonic)

```
h1-6   ( 120- 720 Hz): +24.8 +16.0 +12.4  +9.7  +6.0  -2.5
h7-12  ( 840-1439 Hz):  +1.7  +2.3  -1.8  +3.5  +5.0  +5.8
h13-18 (1559-2159 Hz):  +6.1  +5.8  +5.4  +5.2  +5.7  +6.5
h19-24 (2279-2879 Hz):  +8.6  +3.0  -1.3  -0.3  -0.1  -0.2
h25-30 (2999-3599 Hz):  -1.0  -3.1  -6.7 -13.5  -8.4  -0.0
h31-36 (3719-4318 Hz):  +5.5  +9.8 +13.9 +18.0 +22.6 +26.6
h37-42 (4438-5038 Hz): +32.0 +38.1 +49.7 +49.2 +42.5 +37.9
h43-59 (5158-7077 Hz): +32 to +41
```

Three patterns: the low end is too strong (h1–h5), the F2–F3 valley is too full (h13–h19, about +6 dB), and
everything above about 3.7 kHz is far too loud.

## Finding 1: a stuck spare pole pair (Step 2)

The trained filter's poles at the middle frame (frequency / bandwidth, Hz), from `scripts/step7_diagnostics.py`:

| Order | Poles | 4–6 kHz / 6–8 kHz re peak (target −104 / −110) | LSD | Harmonic RMS < / > 4 kHz |
|---|---|---|---|---|
| 10 (default) | 719/265, 1130/224, 2413/106, **4736/53**, 7955/117 | −74 / −79 | 15.6 | 8.8 / 28.2 |
| 8 | 747/227, 1357/443, 2672/182, 8000/51, 8000/wide | −98 / −103 | 12.4 | 8.5 / 11.2 |
| 12 | 0/wide, 728/276, 1138/227, 2411/73, 3286/57, **6765/55**, 8000/51 | −95 / −86 | 14.4 | 11.0 / 19.4 |

- Order 10 has five resonances for a four-formant target. The fifth settles at 4.7 kHz with the narrowest bandwidth
  γ = 0.99 allows (about 51 Hz). That's the bright stray band in the spectrogram. No pole goes to F4 (3.4 kHz).
- Order 8 has no spare resonance, so the high band is right. But F4 is gone, F2 moves to 1357 Hz and the
  bandwidths widen. It scores best on LSD, which says more about the LSD metric than about the fit.
- Order 12 finds F4 (3286 Hz) and parks its spare poles at 6.8 and 8 kHz instead.

**Conclusion:** changing the order doesn't fix it. The loss does see the error (Finding 3), but gradient descent
can't move a narrow resonance 1–2 kHz across the spectrum, so a spare pole stays wherever it started.

**Recommendation (preferred): a formant-parameterized filter.** Build the all-pole filter as a cascade of
second-order resonators, one per formant:
- Centre frequencies come directly from `f1`–`f4`, the way VocalTract itself works.
- Bandwidths come from a small learned head, or are fixed at about 100 Hz (VocalTract: r = 0.984 at 20 kHz).
- Optionally add one extra real pole or low-order section for overall spectral tilt.

Benefits:
- No free resonances, so there's nowhere for a stray band to come from.
- `f1`–`f4` become identifiable and interpretable (fixes Known Limitation 1, the self-mode finding).
- Stability is just pole radius < 1 per section.
- It matches the project goal of the best *interpretable* synthesizer.

**Interim, if the MLP mapper stays:**
- Initialize reflection coefficients beyond 2 × (number of formants) near 0, so spare sections start disabled.
- Penalize narrow (< ~200 Hz) pole pairs above F4 + 500 Hz.

**Acceptance check:** on `vt_vowel_ah`, no resonance above 4 kHz, and four resonances within 5% of
730 / 1090 / 2440 / 3400 Hz.

## Finding 2: the fixed source tilt (Step 4)

In the target, h1 is about 25 dB below the strongest harmonic (h6, at F1). The spectrum *rises* from 120 Hz to F1,
as expected from VocalTract's glottal pulse. The source is `Σ sin(kφ)/k` (−6 dB/octave), which makes h1 the
strongest harmonic. After training, h1–h5 are still +25 / +16 / +12 / +10 / +6 dB. The only thing the filter can do
is widen F1/F2: 265 / 224 Hz pole bandwidths against about 100 Hz in VocalTract. That also fills in the F2–F3 valley
by about 6 dB.

**Recommendation: make the source tilt learnable, independent of the filter.**
- Use `Σ sin(kφ) / k^α` with a learnable global α (init 1.0), plus a learnable one-pole high-pass on the excitation.
- Make them per-clip learnables, not frame parameters, so the 13-parameter schema stays as it is.
- This is the first piece of the Phase 2 glottal model, which will replace it with a pulse shape behind the same
  interface.

**Acceptance check:** on `vt_vowel_ah`, h1–h5 within ±6 dB, and F1/F2 bandwidths under 150 Hz.

## Finding 3: the loss is not the cause (Step 5, no change)

I suspected the log-magnitude L1 term might ignore the quiet high band because of its floor. Checked: `EPS = 1e-7`,
while the quietest 4–8 kHz STFT magnitudes of `vt_vowel_ah` are about 2e-4 (−102 dB re peak), so the term sees
the +30 to +50 dB excess. The excess persists because of the pole local minimum (Finding 1).

## Corrections to my earlier advice

Two things I recommended earlier turned out to be wrong or already done:

- **"Check the STFT loss's log floor / add a log-spectral loss for the high band."** Retracted (Finding 3).
- **"Loosen bandwidth expansion from γ = 0.98 to 0.99."** Already done in the Step 2 revision (`DEFAULT_GAMMA = 0.99`).
  The wide F1/F2 bandwidths come from the source tilt (Finding 2), not from the γ floor.

## Suggested additions to the Step 7 report

- The trained filter's poles (frequency / bandwidth) at the middle frame. `poles()` in `scripts/step7_diagnostics.py`
  does this in a few lines.
- Harmonic-level RMS error below / above 4 kHz (`vocaltract.analysis.compare.harmonic_levels`).
- **Targets once Findings 1 and 2 are addressed:** ≤ 5 dB below 4 kHz, ≤ 10 dB above, and all four formants
  within 5%.
- On LSD: it rewarded order 8 for losing F4. Read LSD next to the formant and harmonic numbers, not on its own.
