"""
Spectral losses for training the Phase 1 pipeline.

The training signal for the whole chain

    params -> ParamToLPC -> make_excitation -> LPCFilter -> audio

is a multi-resolution STFT loss between predicted and target audio. Comparing
magnitude spectra (rather than raw waveforms) makes the loss insensitive to
phase, which the sine/noise source in Phase 1 cannot match sample-for-sample.

Each scale combines two terms (as in Parallel WaveGAN, Yamamoto et al. 2020):

- **Spectral convergence**: ``||S_t - S_p||_F / ||S_t||_F`` — emphasises the
  large spectral peaks (formants, harmonics).
- **Log-magnitude L1**: ``mean |log S_t - log S_p|`` — emphasises the quiet
  regions between peaks.

Using several FFT sizes trades off time and frequency resolution so no single
window length dominates. Neither ``pred`` nor ``target`` is detached, so the
loss can also back-propagate into a target produced by a trainable model.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

EPS = 1e-7


def stft_magnitude(audio: torch.Tensor, n_fft: int, hop: int, win_length: int) -> torch.Tensor:
    """
    Magnitude STFT with a Hann window.

    Args:
        audio: Signal of shape (B, T).
        n_fft: FFT size.
        hop: Hop length in samples.
        win_length: Window length in samples (<= n_fft).

    Returns:
        Magnitudes ``sqrt(re^2 + im^2 + EPS^2)`` of shape (B, n_fft // 2 + 1, frames).

        The EPS^2 floor (rather than ``abs().clamp_min``) keeps the gradient
        finite: ``abs`` differentiates as z/|z|, and for near-silent audio in
        float32 (denormals around 1e-40) |z| rounds to 0 while z does not.
    """
    window = torch.hann_window(win_length, dtype=audio.dtype, device=audio.device)
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    return torch.sqrt(spec.real.square() + spec.imag.square() + EPS**2)


def spectral_convergence(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
    """Frobenius-norm relative error ``||T - P|| / ||T||`` over the whole batch."""
    return torch.linalg.norm(target_mag - pred_mag) / (torch.linalg.norm(target_mag) + EPS)


def log_magnitude_l1(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference of log magnitudes."""
    return (target_mag.log() - pred_mag.log()).abs().mean()


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss: spectral convergence + log-magnitude L1,
    averaged over several FFT sizes.

    Args:
        ffts: FFT sizes, one per scale.
        hops: Hop length per scale. Defaults to ``n_fft // 4``.
        win_lengths: Window length per scale. Defaults to ``n_fft``.
    """

    def __init__(
        self,
        ffts: Sequence[int] = (512, 1024, 2048),
        hops: Optional[Sequence[int]] = None,
        win_lengths: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.ffts = tuple(ffts)
        self.hops = tuple(hops) if hops is not None else tuple(n // 4 for n in self.ffts)
        self.win_lengths = tuple(win_lengths) if win_lengths is not None else self.ffts

        if not (len(self.ffts) == len(self.hops) == len(self.win_lengths)):
            raise ValueError("ffts, hops and win_lengths must have the same length")
        if any(w > n for w, n in zip(self.win_lengths, self.ffts)):
            raise ValueError("each win_length must be <= its n_fft")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            pred: Predicted audio of shape (B, T) or (T,).
            target: Target audio, same shape as ``pred``.

        Returns:
            Scalar loss tensor.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
        if pred.ndim == 1:
            pred, target = pred.unsqueeze(0), target.unsqueeze(0)
        elif pred.ndim != 2:
            raise ValueError(f"expected (B, T) or (T,) audio, got {tuple(pred.shape)}")

        scale_losses = []
        for n_fft, hop, win_length in zip(self.ffts, self.hops, self.win_lengths):
            pred_mag = stft_magnitude(pred, n_fft, hop, win_length)
            target_mag = stft_magnitude(target, n_fft, hop, win_length)
            scale_losses.append(
                spectral_convergence(pred_mag, target_mag) + log_magnitude_l1(pred_mag, target_mag)
            )
        return torch.stack(scale_losses).mean()
