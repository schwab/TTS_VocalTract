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
│ Parameter → LPC │   small MLP
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
| **Parameter Mapper** | Converts semantic parameters → LPC coefficients | Small MLP, outputs reflection coefficients for stability |
| **LPC Filter** | Time-varying all-pole vocal tract model | `torchlpc.sample_wise_lpc` |
| **Source Model** | Glottal pulse + aspiration noise | Differentiable wavetable / LF-model glottal flow |
| **Decoder (optional)** | Audio → parameters (for training on real speech) | Inverse of the encoder |

## The 13-Parameter Frame

Inherited from the original Propeller VocalTract encoding:

| Parameter | Description |
|---|---|
| Aspiration amplitude | Breathiness / whisper |
| Glottal amplitude | Voicing strength |
| Glottal pitch | Fundamental frequency |
| F1–F4 | Formant frequencies |
| Fricative amplitude | Sibilant / fricative energy |
| Nasal formant | Anti-resonator for nasals |
| ... | (full table TBD) |

Each frame is emitted at ~50 Hz (20 ms hop), matching the original Propeller update rate.

## Key Technical Decisions

### Why LPC instead of CORDIC?

The original VocalTract uses **CORDIC rotations** for its formant resonators — efficient on the Propeller, but **non-differentiable**. We replace the CORDIC resonator with an equivalent **second-order IIR / LPC filter**, which:

- Has an efficient, closed-form gradient (via `torchlpc`)
- Preserves the all-pole formant topology
- Keeps F1–F4 as interpretable poles

### Why "learned mapping" instead of "hard conversion"?

Rather than hand-designing the mapping from the 13 parameters to LPC coefficients, a small MLP learns it. Evidence from **GOLF** (Yu et al.) shows that LPC filters spontaneously learn spectral envelopes aligned with formants — so the network naturally discovers the right structure.

### Stability

LPC coefficients can produce unstable filters if unconstrained. The mapper outputs **reflection coefficients** (PARCOR), which are guaranteed stable under the Levinson–Durbin recursion.

## Loss Functions

Training uses a combination of:

1. **Multi-scale STFT loss** — spectral envelope fidelity
2. **Formant tracking loss** — weakly supervises F1–F4 alignment (using LPC analysis on the synthesized audio)
3. **Adversarial loss (optional)** — multi-resolution discriminator for perceptual quality
4. **Parameter reconstruction loss** — when training with a paired encoder/decoder

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

## Roadmap

- [ ] **Phase 1** — Minimal differentiable LPC pipeline (random audio → params → LPC → reconstruction)
- [ ] **Phase 2** — Glottal source model (differentiable LF / wavetable)
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

🚧 **Early design phase.** No working implementation yet. Contributions and discussion welcome.

## License

TBD

---

Want me to adjust anything? A few things I'd want to confirm before you commit:

1. **License** — MIT? Apache 2.0?
2. **Repo name / branding** — keep "TTS_VocalTract", or something like "NeuralVocalTract"?
3. **The 13-parameter table** — I left the last few entries as placeholders. If you have the exact parameter list from the original VocalTract, I can fill it in.
4. **Tone** — I aimed for technical-but-approachable. Want it more academic, or more casual/hacker-style?
