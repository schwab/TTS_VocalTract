# Sample recordings

Short speech clips used to try `vocaltract.analysis.resynth` (level A analysis/resynthesis).

| File | Speaker | Source |
|---|---|---|
| `198-209-0000.ogg` | Heather Barnett (female) | LibriSpeech 198-209-0000. librosa's index calls it *Sense and Sensibility* ch. 18, but the text is Jane Austen's *Northanger Abbey* (Catherine, Mrs. Allen, Miss Tilney) |
| `5703-47212-0000.ogg` | Garth Comira (male) | LibriSpeech 5703-47212-0000, *The Ashiel Mystery* ch. XVI |

Both come from LibriSpeech (V. Panayotov et al., derived from LibriVox public-domain audiobooks), licensed
**CC BY 4.0**. Fetched via librosa's example data (`github.com/librosa/data`, `audio/<id>.hq.ogg`).
16 kHz mono, about 14 s each.

## Reference transcripts

`<id>.txt` next to each clip is the reference text for the Whisper word-error-rate metric
(`vocaltract.analysis.intelligibility`). `198-209-0000.txt` follows the public-domain book text.
`5703-47212-0000.txt` is Whisper medium.en's transcript with "lock" corrected to "loch"; the
Scottish place names are best guesses, so expect a couple of name errors on every render,
including the original recording.
