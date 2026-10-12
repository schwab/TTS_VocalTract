# TTS_VocalTract

A differentiable, LPC-based neural vocoder inspired by the Parallax Propeller's VocalTract speech synthesizer.

## Overview

The Parallax Propeller's **VocalTract** object (circa 2006–2015) was a remarkable feat of embedded engineering: a real-time, phoneme-driven speech synthesizer running on a bare-metal 8-core microcontroller, controlled by a **13-byte parameter frame** encoding glottal excitation, aspiration, four formant frequencies, and fricative/nasal characteristics.

This project asks a simple question:

> What if we took that 13-parameter encoding, made the synthesizer **fully differentiable**, and trained it end-to-end with modern neural networks?

The result is a **neural vocal tract** — a source-filter synthesizer whose latent space is physically interpretable, controllable, and trainable via backpropagation.

## Motivation

Modern neural TTS systems (Tacotron, VITS, etc.) are powerful but opaque. Their latent spaces are entangled and offer little direct control over vocal characteristics like breathiness, vibrato, or formant structure.

The original VocalTract parameterization offers the opposite: a **compact, interpretable, physically grounded** control signal. This project aims to bridge the two worlds:

- **Interpretability** of a classic source-filter model
- **Expressiveness** of a learned neural codec
- **Controllability** over emotion, emphasis, and speaker identity via direct parameter manipulation

## Architecture

```
Text / Audio
      │
      ▼
┌─────────────────┐
│  Encoder (NN)   │   CNN / Transformer
└────────┬────────┘
         │  13-parameter frame @ 50 Hz
         ▼
┌─────────────────┐
│ Parameter → LPC │   one resonator per formant (f1–f4)
│    Mapper       │
└────────┬────────┘
         │  time-varying LPC coefficients
         ▼
┌─────────────────┐
│ Differentiable  │   torchlpc / custom kernels
│  LPC Filter     │
└────────┬────────┘
         │
         ▼
   Glottal Source + Aspiration Noise
         │
         ▼
       Audio
```

### Components

| Component | Role | Implementation |
|---|---|---|
| **Encoder** | Maps audio (or text) to the 13-parameter frame | CNN or Transformer |
| **Parameter Mapper** | Converts semantic parameters → LPC coefficients | `FormantToLPC`: a pole pair at each of f1–f4 with learned bandwidths, plus learned tilt poles (`ParamToLPC`, the original MLP, is still available) |
| **LPC Filter** | Time-varying all-pole vocal tract model | `torchlpc.sample_wise_lpc` |
| **Source Model** | Glottal pulse + aspiration noise | VocalTract's own glottal pulse, rendered at 20 kHz and resampled differentiably to 16 kHz |
| **Decoder (optional)** | Audio → parameters (for training on real speech) | Inverse of the encoder |

## The 13-Parameter Frame

Inherited from the original Propeller VocalTract encoding:

| Name | Parameter | Description |
|---|---|---|
| `aa` | Aspiration amplitude | Breathiness / whisper (enters before the formants) |
| `ga` | Glottal amplitude | Voicing strength |
| `gp` | Glottal pitch | Fundamental frequency, log scale: 1/48 octave per unit, 100 = 110 Hz |
| `vp` | Vibrato pitch | Vibrato depth (`vp`/96 octaves of swing) |
| `vr` | Vibrato rate | Vibrato frequency (0.0763 Hz per unit) |
| `f1`–`f4` | Formant frequencies | 19.53 Hz per unit |
| `na` | Nasal amplitude | Depth of the nasal anti-resonance |
| `nf` | Nasal frequency | Anti-resonance frequency (19.53 Hz per unit) |
| `fa` | Frication amplitude | Sibilant / fricative energy (added after the formants) |
| `ff` | Frication frequency | Centre of the frication noise (39.06 Hz per unit) |

Each value is one byte on the Propeller and is normalized to [0, 1] here (`vocaltract.params`). Phase 1
frames are 20 ms apart (50 Hz).

## Key Technical Decisions

### Why LPC instead of CORDIC?

The original VocalTract uses **CORDIC rotations** for its formant resonators — efficient on the Propeller, but **non-differentiable**. We replace the CORDIC resonator with an equivalent **second-order IIR / LPC filter**, which:

- Has an efficient, closed-form gradient (via `torchlpc`)
- Preserves the all-pole formant topology
- Keeps F1–F4 as interpretable poles

### Why a formant mapper instead of a learned MLP?

The original plan was for a small MLP to learn the mapping from the 13 parameters to LPC coefficients, following
**GOLF** (Yu et al.), where LPC filters learn formant-aligned envelopes. Phase 1 showed that this mapping is not
identifiable. The MLP fitted the spectrum while the formant parameters drifted away from the truth, and on an
/a/→/i/ glide it held one compromise vowel. `FormantToLPC` places one resonator at each of f1–f4, as VocalTract
does, and learns only what the frame doesn't specify: four bandwidths and the spectral tilt. Formant parameters
are then recoverable from audio (perturbed by ~135 Hz, they return to within ~10–50 Hz).

### Stability

LPC coefficients can produce unstable filters if unconstrained. `FormantToLPC` builds the filter from its poles:
each bandwidth has a 40 Hz floor and each tilt pole is bounded, so every pole radius stays below 1. (The MLP mapper
outputs **reflection coefficients** (PARCOR), which are stable under the Levinson–Durbin recursion.)

## Loss Functions

Training uses a combination of:

1. **Multi-scale STFT loss** — spectral envelope fidelity
2. **Formant tracking loss** — weakly supervises F1–F4 alignment (using LPC analysis on the synthesized audio)
3. **Adversarial loss (optional)** — multi-resolution discriminator for perceptual quality
4. **Parameter reconstruction loss** — when training with a paired encoder/decoder

## Phase 1: Minimal Differentiable LPC Pipeline

Phase 1 is a differentiable model of the VocalTract voice path. 13-parameter frames go through the engine's glottal pulse plus aspiration noise, then one resonator per formant, to 16 kHz audio. Gradients reach every parameter, and a 1 s clip can be fitted by gradient descent on the frames. The full specification and results are in [`Phase_1.md`](Phase_1.md) and [`docs/reviews/phase1_completion_review.md`](docs/reviews/phase1_completion_review.md).

**Install** (Python ≥ 3.10, CPU only; `torchlpc` builds a C++ extension and needs the `python3-dev` headers):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install --no-deps -e .
pytest -m "not slow"     # ~35 s; `pytest -m slow` adds two full overfit runs (~2 min)
```

The extra index fetches the CPU build of PyTorch, which is much smaller than the default CUDA build. Leave it out to use a GPU build.

**Run:**

```bash
python -m vocaltract.phase1.smoke_test     # one forward/backward pass; prints shapes and the gradient norm
python -m vocaltract.phase1.overfit_demo   # fits assets/phase1 vt_vowel_ah; writes out/phase1_overfit*.wav
python -m vocaltract.phase1.overfit_demo --target vt_glide_ah_ee --target-mode self   # parameter recovery
```

The demo targets are in `assets/phase1` (see its README): VocalTract renders, a synthetic vowel and two speech excerpts.

- `--target-mode wav` (the default) fits the WAV, starting from the clip's `.npz` parameters.
- `--target-mode self` renders the target with the pipeline itself, perturbs the formants, and checks that the fit moves them back toward the truth.
- `python scripts/overfit_report.py out/` summarises a set of runs: harmonic-level error, plus Whisper WER with `--asr`.

**Status.**
- On the VocalTract targets, harmonic-level error after fitting is 0.2–0.7 dB below 4 kHz.
- Self mode recovers formants perturbed by ~135 Hz to within ~10–50 Hz.
- On real speech the fit plateaus at 4–6 dB.

Known limitations, which Phase 2 addresses:
- Frication, nasals and vibrato (`fa`, `ff`, `na`, `nf`, `vp`, `vr`) are not modelled yet.
- Pitch is only learnable locally (within ~6 cents), so every fit starts from given parameters.
- Real voices are not shaped like VocalTract's pulse.

The glottal source originally planned for Phase 2 already exists: the pipeline uses the engine's own pulse.

## Reference Engine

`vocaltract.reference` is a bit-faithful, non-differentiable Python port of the original
`spin/VocalTract.spin` assembly engine. It uses the same 32-bit integer math, ROM tables,
CORDIC resonators and frame interpolation. It is the ground truth for what the Propeller
plays from a frame sequence. `reference/spin.py` holds the shared pieces for porting
more Spin programs: a .spin CON/DAT parser, Spin's `?rnd`, the VocalTract object API,
and the render/mix/CLI.

```bash
pip install numba soundfile    # numba is optional but ~100x faster
python -m vocaltract.reference.singing_demo_seven         # out/singing_demo_seven.wav (faithful timing)
python -m vocaltract.reference.singing_demo_seven --sync  # parts in lockstep
python -m vocaltract.reference.singing_demo               # out/singing_demo.wav (held vowels)
```

### Analysis / resynthesis (level A)

`vocaltract.analysis.resynth` maps a recording onto VocalTract frames with classic speech analysis
(Praat pitch and formants, band levels, no ML), calibrates the source levels against the reference
engine, and renders the result. It's a baseline for how much of a real voice fits in 13 bytes, and
a starting point for gradient fitting.

```bash
pip install praat-parselmouth matplotlib
python -m vocaltract.analysis.resynth assets/recordings/198-209-0000.ogg --spin
# out/resynth/: _resynth.wav, _ab.wav (original then resynthesis), _frames.csv, .png, _player.spin
```

The `_player.spin` file plays the same frames on a real Propeller (P10/P11, needs `VocalTract.spin`).

**Intelligibility metric.** The report includes the Whisper word error rate (WER) of the resynthesis
next to the WER of the original recording (the floor), scored against `<clip>.txt` when present
(`--asr none` to skip, `--asr base.en` for a faster model). Spectral metrics (log-mel distance, STOI)
proved misleading on this project, so WER plus listening is the yardstick. To score any WAV:
`python -m vocaltract.analysis.intelligibility file.wav --ref-file transcript.txt`.

**Side-by-side comparison.** Each run also writes `<name>_compare.html`: a self-contained page with
linked spectrograms of the original and the resynthesis on identical settings. It has a formants view
(5 ms window) and a harmonics view (35 ms), a live dB-floor slider, synced playback (keys 1/2 switch
files at the same moment), overlays of the frames, Praat tracks and Whisper word timings, pitch and
source lanes, and a difference pane. Compare any files directly:

```bash
python -m vocaltract.analysis.compare a.wav b.wav --frames out/resynth/x_frames.csv --words \
    --sv-dir out/sv_layers    # optional: overlays as Sonic Visualiser CSV layers
```

Views are shareable through the URL, e.g. `x_compare.html#t0=1.6&t1=3.6&res=narrow&diff=1`.

**Folder browser.** To pick files from the page instead of the command line, serve a folder:

```bash
python -m vocaltract.analysis.compare --serve .        # or --serve out/, assets/phase1/, ...
```

This starts a local server (http://127.0.0.1:8765/, localhost only) and opens the page. Each track has a
menu of every audio file under the folder (wav/flac/ogg/mp3, grouped by directory). There are buttons to
add or remove tracks, a **Frames file** menu for `*_frames.csv` overlays, and a per-track **words** button
(Whisper timings, plus WER when a `<stem>.txt` transcript sits next to the file). Spectrograms are computed
when you pick a file and are cached. The selection is kept in the URL (`#files=a.wav,b.wav&framesfile=...`),
so reloading or bookmarking restores it. **↻ rescan** picks up files written since the page loaded.

The **harmonic levels** pane (both modes) plots each track's harmonic envelope: levels at k·f0 along the Praat
pitch track, averaged over the voiced frames in view, with the difference vs track 1 and its RMS error below and
above 4 kHz. Zoom in time to analyse a single vowel.

## Roadmap

- [x] **Phase 1** — Minimal differentiable LPC pipeline (random audio → params → LPC → reconstruction)
- [ ] **Phase 2** — Glottal source model (differentiable LF / wavetable); proposed rescope: complete source model and
  utterance fitting, measured by WER (`docs/reviews/phase1_completion_review.md`)
- [ ] **Phase 3** — Train audio → parameters encoder on real speech
- [ ] **Phase 4** — Formant tracking loss + interpretability analysis
- [ ] **Phase 5** — Text → parameters (TTS)
- [ ] **Phase 6** — Speaker identity / emotion control via parameter modulation
- [ ] **Phase 7** — Export to embedded / real-time inference

## References

- **Parallax Propeller VocalTract** — Chip Gracey, *Programming and Customizing the Multicore Propeller Microcontroller*, Ch. 12
- **GOLF** — Yu et al., *Differentiable Glottal-Flow LPC Filter for Singing Voice Synthesis*
  - GitHub: `yoyololicon/golf`
- **DDSP** — Engel et al., *DDSP: Differentiable Digital Signal Processing*
- **torchlpc** — `yoyololicon/torchlpc`, efficient differentiable IIR/LPC filtering in PyTorch

## Status

🚧 **Phase 1 complete; Phase 2 in planning.** Contributions and discussion welcome.

## License

TBD
