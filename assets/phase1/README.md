# Phase 1 Step 7 overfit targets

Seven 1-second targets for `vocaltract.phase1.overfit_demo`, ordered by difficulty. Each is a
16 kHz mono WAV of exactly 16000 samples (peak -6 dBFS), plus an `.npz` of parameters on the
Phase 1 frame grid (hop 320 → 50 frames). `manifest.json` lists them all.

| # | Target | Params | What it is |
|---|---|---|---|
| 1 | `synthetic_vowel_ah` | ground truth | Classic source-filter /a/ (not VocalTract): f0 120 Hz, formants 730/1090/2440/3400 Hz with natural bandwidths |
| 2 | `vt_vowel_ah` | ground truth | VocalTract reference engine, steady /a/ at 120 Hz |
| 3 | `vt_glide_ah_ee` | ground truth | VocalTract, /a/ → /i/ glide with a 110 → 150 Hz pitch rise |
| 3 | `vt_sung_oo_vibrato` | ground truth | VocalTract, sung /u/ at 196 Hz with 5.3 Hz vibrato (±0.75 semitone) |
| 4 | `vt_seven` | ground truth | One "seven" from Chip Gracey's SingingDemoSeven (tenor): includes frication and a nasal |
| 5 | `speech_male` | analysis | "during the summer months", LibriSpeech male reader (f0 ~80 Hz) |
| 5 | `speech_female` | analysis | "(Cather)ine, the next morning", LibriSpeech female reader (f0 ~210 Hz) |

## Loading

```python
import numpy as np, soundfile as sf, torch

audio, sr = sf.read("assets/phase1/vt_vowel_ah.wav")          # (16000,), 16000
d = np.load("assets/phase1/vt_vowel_ah.npz")
target = torch.tensor(audio, dtype=torch.float32)[None]          # (1, 16000)
params = torch.tensor(d["params"])[None]                         # (1, 50, 13), normalized [0, 1]
```

`.npz` contents:

| Key | Meaning |
|---|---|
| `params` | (50, 13) float32, `raw / 255` in `PARAM_NAMES` order (aa, ga, gp, vp, vr, f1-f4, na, nf, fa, ff), the normalization `vocaltract.params.denormalize` expects |
| `params_raw` | (50, 13) the same in VocalTract byte units (fractional where interpolated) |
| `frame_times` | (50,) frame centres in seconds, `(t + 0.5) * 0.02`, matching `source.upsample_frames` (`align_corners=False`) |
| `kind` | `ground_truth` or `analysis` |
| `sample_rate`, `hop`, `param_names` | 16000, 320, names |

## Things to know before using them

- **What "ground truth" means.** The parameters are exact *VocalTract* values: the engine's
  piecewise-linear parameter trajectory, sampled at frame centres. For the synthetic vowel they
  are the physical values used to build it. The Phase 1 pipeline only interprets `gp`, `ga` and
  `aa` directly (sine + noise source). Formants pass through the learned `ParamToLPC` mapper, which
  is untrained in Phase 1. So feeding these params into Phase 1 will **not** reproduce the WAV.
  Use them as an initialization, or to check whether optimized `gp` lands near the truth.
- **Amplitude units differ.** The WAVs are peak-normalized to 0.5. `ga`/`aa` are in VocalTract
  units and don't correspond to Phase 1 output levels.
- **Option (a) of Step 7** (synthesize the target with the same pipeline): render it from one of
  these `params` files with a fixed-seed mapper. That's a self-consistent target where exact
  recovery is possible.
- **Option (b)** (load a WAV): start with targets 1–2. The Phase 1 sine source can't produce
  vibrato (`vp`/`vr`), frication (`fa`/`ff`) or nasals (`na`/`nf`), so 3–5 have a loss floor by
  design. They show what Phase 2+ needs to add.
- **Noise.** `make_excitation` draws fresh noise unless given a `generator`. Use a fixed generator
  per step (or `aa = 0`), or the waveform L1 term will have an irreducible floor.
- **Speech params are approximate.** They come from level A analysis (`vocaltract.analysis.resynth`,
  Praat pitch/formants plus engine-calibrated levels). They're a reasonable initialization, not
  ground truth.

## Checks done when generating

- Every file is 16000 samples at 16 kHz. Every `params` array is (50, 13) within [0, 1].
- Praat pitch on each WAV matches its `gp` within 0–10 cents (median). The vibrato target reads
  55 cents because `gp` holds the centre pitch and the swing is in `vp`/`vr`.
- Whisper hears "during the summer months" and "in the next morning" in the speech excerpts.
- All seven run through the Step 6 pipeline (`ParamToLPC` → `make_excitation` → `LPCFilter`,
  `MultiScaleSTFTLoss` + L1). Output is finite, and the loss falls within 30 Adam steps on 6 of 7
  (the synthetic vowel stayed flat at lr 1e-2 with the mapper frozen).

## Regenerating

```bash
python scripts/make_phase1_assets.py    # needs the analysis extras and openai-whisper
```

Sources: VocalTract targets come from `vocaltract.reference` (MIT-licensed Parallax engine,
ported). Speech excerpts come from `assets/recordings/` (LibriSpeech, CC BY 4.0, see the README there).
