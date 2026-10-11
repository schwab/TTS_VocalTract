"""
Intelligibility metric: word error rate (WER) of Whisper speech recognition.

"Can a listener make out the words?" is the question that matters for a TTS
vocalizer, and spectral metrics don't answer it. On this project log-mel
distance barely moved for changes that were obvious by ear, and STOI ranked
the two test voices backwards. An off-the-shelf recognizer transcribing the
synthesized audio, scored against the reference text, tracks intelligibility
far better.

Always read the synthesized WER next to the WER of the *original recording*
with the same model: that is the floor (recognizer mistakes, proper nouns,
imperfect references), and the gap between the two is what the synthesizer
loses.

Caveat: Whisper's language model can guess words from context, so WER is an
optimistic measure of intelligibility for long, predictable sentences. It is
still the best single number here; pair it with listening checks.

Usage::

    python -m vocaltract.analysis.intelligibility out/resynth/x_resynth.wav \\
        --ref-file assets/recordings/x.txt

Requires ``openai-whisper`` and ``jiwer``. Models download on first use to
``~/.cache/whisper``.
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "small.en"
ASR_RATE = 16_000


@dataclass
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis: str

    def summary(self) -> str:
        return (f"WER {self.wer:.0%} ({self.substitutions} sub, {self.deletions} del, "
                f"{self.insertions} ins / {self.reference_words} words)")


@lru_cache(maxsize=2)
def load_model(name: str = DEFAULT_MODEL):
    import whisper

    return whisper.load_model(name, device="cpu")


@lru_cache(maxsize=1)
def _normalizer():
    from whisper.normalizers import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    """Whisper's English normalizer: case, punctuation, spelling variants, numbers."""
    return _normalizer()(text)


def transcribe(audio: np.ndarray, sr: int, model: str = DEFAULT_MODEL) -> str:
    """Greedy, context-free English transcription of a mono float signal."""
    from scipy import signal

    x = np.asarray(audio, dtype=np.float64)
    if sr != ASR_RATE:
        x = signal.resample_poly(x, ASR_RATE, sr)
    peak = np.abs(x).max()
    if peak > 0:
        x = x * (0.9 / peak)
    result = load_model(model).transcribe(
        x.astype(np.float32), language="en", temperature=0.0,
        condition_on_previous_text=False, fp16=False)
    return result["text"].strip()


def transcribe_words(audio: np.ndarray, sr: int,
                     model: str = DEFAULT_MODEL) -> tuple[str, list[dict]]:
    """Like :func:`transcribe`, plus word timings ``[{"word", "start", "end"}, ...]`` (seconds)."""
    from scipy import signal

    x = np.asarray(audio, dtype=np.float64)
    if sr != ASR_RATE:
        x = signal.resample_poly(x, ASR_RATE, sr)
    peak = np.abs(x).max()
    if peak > 0:
        x = x * (0.9 / peak)
    result = load_model(model).transcribe(
        x.astype(np.float32), language="en", temperature=0.0, word_timestamps=True,
        condition_on_previous_text=False, fp16=False)
    words = [{"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
             for seg in result["segments"] for w in seg.get("words", [])]
    return result["text"].strip(), words


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    import jiwer

    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref.split():
        raise ValueError("empty reference transcript")
    out = jiwer.process_words(ref, hyp if hyp.split() else "<empty>")
    return WerResult(wer=out.wer, substitutions=out.substitutions, deletions=out.deletions,
                     insertions=out.insertions, reference_words=len(ref.split()),
                     hypothesis=hypothesis)


def score(audio: np.ndarray, sr: int, reference: str, model: str = DEFAULT_MODEL) -> WerResult:
    return word_error_rate(reference, transcribe(audio, sr, model))


def reference_for(audio_path: Path) -> str | None:
    """The ``<stem>.txt`` transcript next to an audio file, if there is one."""
    txt = audio_path.with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Whisper word error rate of audio files")
    ap.add_argument("inputs", nargs="+", type=Path)
    ref = ap.add_mutually_exclusive_group()
    ref.add_argument("--ref", help="reference transcript text")
    ref.add_argument("--ref-file", type=Path, help="reference transcript file")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model (default {DEFAULT_MODEL})")
    args = ap.parse_args(argv)

    import soundfile as sf

    for path in args.inputs:
        reference = (args.ref or (args.ref_file.read_text(encoding="utf-8") if args.ref_file else None)
                     or reference_for(path))
        audio, sr = sf.read(path, always_2d=True)
        hyp = transcribe(audio.mean(axis=1), sr, args.model)
        print(f"{path}: {hyp}")
        if reference:
            print(f"  {word_error_rate(reference, hyp).summary()}")


if __name__ == "__main__":
    main()
