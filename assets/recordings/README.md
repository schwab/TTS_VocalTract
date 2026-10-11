# Sample recordings

Ten read-speech sentences (5 female, 5 male, about 12–15 s each, 16 kHz mono) used for level A
analysis/resynthesis (`vocaltract.analysis.resynth`) and as the WER evaluation set for Phase 2.
`manifest.json` lists every clip with speaker, gender, duration, median pitch, word count, the Whisper
(small.en) WER of the **original** recording, and the WER of the level A resynthesis.

| File | Speaker | Median f0 | Source |
|---|---|---|---|
| `198-209-0000.ogg` | Heather Barnett (F) | 212 Hz | LibriSpeech 198-209-0000 via librosa's example data. librosa's index calls it *Sense and Sensibility* ch. 18, but the text is Jane Austen's *Northanger Abbey* |
| `5703-47212-0000.ogg` | Garth Comira (M) | 83 Hz | LibriSpeech 5703-47212-0000 via librosa's example data, *The Ashiel Mystery* ch. XVI |
| `2094-142345-0006.flac` | amycsj (F) | 194 Hz | LibriSpeech test-clean |
| `2961-960-0019.flac` | Leni (F) | 167 Hz | LibriSpeech test-clean |
| `3575-170457-0015.flac` | supergirl (F) | 199 Hz | LibriSpeech test-clean |
| `8463-294828-0035.flac` | Michele Fry (F) | 145 Hz | LibriSpeech test-clean |
| `7729-102255-0015.flac` | Tim Bower (M) | 91 Hz | LibriSpeech test-clean |
| `8230-279154-0013.flac` | David Jenkins (M) | 105 Hz | LibriSpeech test-clean |
| `7127-75946-0002.flac` | Bill Kneeland (M) | 125 Hz | LibriSpeech test-clean |
| `6930-75918-0001.flac` | Nolan Fout (M) | 156 Hz | LibriSpeech test-clean |

All come from LibriSpeech (V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on
public domain audio books", ICASSP 2015; derived from LibriVox public-domain recordings), licensed **CC BY 4.0**.
The test-clean clips come from `https://www.openslr.org/resources/12/test-clean.tar.gz` and are stored unmodified
as FLAC.

**How the test-clean clips were chosen:**
- All 4 female speakers and 4 of the 8 male speakers in the first 12 speakers of test-clean. The men were chosen
  to span pitch (91–156 Hz) alongside the existing 83 Hz voice.
- Per speaker, the 10–15 s utterance with the lowest Whisper WER on the original recording (ties broken by
  word count). All eight score 0%, so a resynthesis's WER is almost entirely what the synthesizer loses.

**Level A baseline** (`level_a_wer` in the manifest): corpus WER 36% (144 errors / 400 words), per clip 3–47%,
except `6930-75918-0001` at 133%. That's a fast, breathy reader with a hiss floor, which level A renders mostly as
noise. It's kept on purpose as the hard case. Details: `docs/reviews/phase1_completion_review.md`.

## Reference transcripts

`<id>.txt` next to each clip is the reference text for the Whisper word-error-rate metric
(`vocaltract.analysis.intelligibility`). For the test-clean clips it is LibriSpeech's official transcript,
lower-cased. `198-209-0000.txt` follows the public-domain book text. `5703-47212-0000.txt` is Whisper medium.en's
transcript with "lock" corrected to "loch". Its Scottish place names are best guesses, so expect a couple of name
errors on every render, including the original recording.
