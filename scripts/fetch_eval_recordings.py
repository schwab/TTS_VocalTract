"""
Re-download the eight LibriSpeech test-clean clips of the evaluation set into assets/recordings/.

Streams ``test-clean.tar.gz`` from openslr (346 MB; stops once every clip is found, so usually well under half of
it) and writes ``<id>.flac`` (unmodified) and ``<id>.txt`` (official transcript, lower-cased). The selection
rationale is in assets/recordings/README.md; manifest.json holds the per-clip metadata and WERs.

Usage::

    python scripts/fetch_eval_recordings.py            # skips clips that already exist
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
CLIPS = ["2094-142345-0006", "2961-960-0019", "3575-170457-0015", "8463-294828-0035",
         "7729-102255-0015", "8230-279154-0013", "7127-75946-0002", "6930-75918-0001"]
DEST = Path(__file__).resolve().parents[1] / "assets" / "recordings"


def main() -> int:
    wanted = {c for c in CLIPS if not (DEST / f"{c}.flac").exists() or not (DEST / f"{c}.txt").exists()}
    if not wanted:
        print("all clips present")
        return 0
    chapters = {c.rsplit("-", 1)[0] for c in wanted}
    audio, text = set(), set()
    with urllib.request.urlopen(URL, timeout=60) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for m in tf:
            name = m.name.split("/")[-1]
            if name.endswith(".flac") and name[:-5] in wanted:
                (DEST / name).write_bytes(tf.extractfile(m).read())
                audio.add(name[:-5])
            elif name.endswith(".trans.txt") and name.split(".")[0] in chapters:
                for line in tf.extractfile(m).read().decode().splitlines():
                    uid, words = line.split(" ", 1)
                    if uid in wanted:
                        (DEST / f"{uid}.txt").write_text(words.lower() + "\n", encoding="utf-8")
                        text.add(uid)
            if audio == wanted and text == wanted:
                break
    missing = wanted - (audio & text)
    print(f"fetched {len(audio & text)} clips" + (f"; missing {sorted(missing)}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
