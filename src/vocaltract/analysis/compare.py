"""
Side-by-side spectrogram comparison of audio files, in the browser.

Two ways to run it:

- **static page**: one self-contained HTML file (no server, no network) for a
  fixed list of files. ``vocaltract.analysis.resynth`` writes one of these per
  input (``<name>_compare.html``).
- **folder browser** (``--serve DIR``): a small local web server. Every track
  has a menu of the audio files under ``DIR`` (searched recursively), so you
  can load, swap, add and remove tracks from the page. Spectrograms and Praat
  tracks are computed when you pick a file. A frames menu lists
  ``*_frames.csv`` files, and each track has a **words** button (Whisper
  timings, plus WER when a ``<stem>.txt`` transcript sits next to the file).
  The selection is kept in the URL, so reloading or bookmarking restores it.

Both show:

- **linked spectrograms**, one pane per file, on identical settings: the same
  frequency range, window and dB floor, each normalized to its own peak so a
  gain difference doesn't read as a spectral one
- two analysis views: **Formants** (5 ms window: smears harmonics, shows the
  vocal-tract envelope) and **Harmonics** (35 ms window: resolves pitch
  harmonics, shows formant/harmonic interaction)
- a **dB floor slider**, applied live (spectrograms are stored as 8-bit dB)
- **synced playback**: play any file with a moving playhead; keys 1-9 switch
  files at the same time position for instant A/B
- overlays: VocalTract **frames** (formants on the spectrograms, pitch and
  source amplitudes in their own lanes), **Praat** pitch/formant tracks per
  file (formants shown on voiced frames only), and **Whisper word timings**
- **URL options** for shareable views:
  ``#t0=2&t1=4&res=narrow&floor=-60&fmax=5000&diff=1&praat=1&frames=0&words=0``
  (folder browser: also ``files=a.wav,b.wav&framesfile=x_frames.csv``)
- a **difference** pane (track 2 minus track 1, in dB)

Usage::

    python -m vocaltract.analysis.compare original.wav resynth.wav \\
        --frames out/resynth/x_frames.csv --words -o out/compare.html

    python -m vocaltract.analysis.compare --serve out/     # folder browser
    python -m vocaltract.analysis.compare --serve . assets/phase1/vt_vowel_ah.wav

``--sv-dir`` also exports the overlays as CSV layers for Sonic Visualiser
(File > Import Annotation Layer).
"""

import argparse
import base64
import io
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..params import FORMANT_SCALE, PARAM_NAMES

# name -> (window seconds, hop seconds, frequency rows)
RESOLUTIONS = {"wide": (0.005, 0.002, 256), "narrow": (0.035, 0.005, 512)}
FLOOR_DB = -100.0          # stored range; the page's slider picks the display floor
DIFF_RANGE_DB = 40.0       # difference pane spans +/- this
DIFF_QUIET_DB = -70.0      # silent in both files -> shown as equal
MAX_COLUMNS = 12_000       # longer files get a coarser hop
SERVE_FMAX = 8000.0        # folder browser's top frequency (16 kHz Nyquist)
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"}
_SKIP_DIRS = {"__pycache__", "node_modules", "venv", "site-packages"}
GP_INDEX, GA_INDEX, AA_INDEX, FA_INDEX = (PARAM_NAMES.index(n) for n in ("gp", "ga", "aa", "fa"))
F1_INDEX = PARAM_NAMES.index("f1")


@dataclass
class Clip:
    name: str
    audio: np.ndarray      # mono float
    sr: int
    note: str = ""         # shown next to the name, e.g. "WER 43%"

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


def load_clip(path: Path, name: str | None = None, start: float = 0.0,
              duration: float | None = None) -> Clip:
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=True)
    audio = audio.mean(axis=1)
    lo = int(start * sr)
    hi = len(audio) if duration is None else lo + int(duration * sr)
    return Clip(name or path.stem, audio[lo:hi], sr)


# ---------------------------------------------------------------------------
# spectrograms
# ---------------------------------------------------------------------------

def _frequency_weights(bin_freqs: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Triangular weights mapping FFT bins onto display rows (averages when downsampling)."""
    spacing = max(rows[1] - rows[0], bin_freqs[1] - bin_freqs[0])
    w = np.clip(1 - np.abs(rows[:, None] - bin_freqs[None, :]) / spacing, 0, None)
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)


def spectrogram(audio: np.ndarray, sr: int, window_s: float, hop_s: float, fmax: float,
                n_cols: int, n_rows: int) -> np.ndarray:
    """Power spectrogram in dB re its own peak, ``(n_rows, n_cols)``, low frequency first.

    Column k is centred at ``k * hop_s``; columns past the end of the audio
    (and rows above the file's Nyquist) sit at the floor.
    """
    x = np.asarray(audio, dtype=np.float64)
    nper = max(16, int(round(window_s * sr)))
    nfft = 1 << int(np.ceil(np.log2(max(2 * nper, 256))))
    centres = np.round(np.arange(n_cols) * hop_s * sr).astype(np.int64)
    padded = np.pad(x, (nper, nper + int(n_cols * hop_s * sr) + 1))
    idx = centres[:, None] + np.arange(nper)[None, :] + nper - nper // 2
    frames = padded[idx] * np.hanning(nper)[None, :]
    power = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2              # (n_cols, bins)
    rows = np.linspace(0, fmax, n_rows)
    mapped = power @ _frequency_weights(np.fft.rfftfreq(nfft, 1 / sr), rows).T
    db = 10 * np.log10(mapped.T + 1e-20)
    db -= db.max()
    return np.maximum(db, FLOOR_DB)


def _png_b64(gray: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.flipud(gray), mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_db(db: np.ndarray) -> str:
    """8-bit greyscale PNG: 0 = FLOOR_DB, 255 = 0 dB; top row = highest frequency."""
    return _png_b64(np.round((db - FLOOR_DB) / -FLOOR_DB * 255).astype(np.uint8))


def clip_spectrograms(clip: Clip, fmax: float) -> dict:
    """Both views of one clip on its own time grid: ``{res: {hop, cols, rows, png}}``."""
    out = {}
    for res, (window, hop, n_rows) in RESOLUTIONS.items():
        hop = max(hop, clip.duration / MAX_COLUMNS)
        n_cols = int(np.ceil(clip.duration / hop)) + 1
        db = spectrogram(clip.audio, clip.sr, window, hop, fmax, n_cols, n_rows)
        out[res] = {"hop": hop, "cols": n_cols, "rows": n_rows, "png": encode_db(db)}
    return out


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """16-bit WAV, peak-normalized to 0.9."""
    import soundfile as sf

    x = np.asarray(audio, dtype=np.float64)
    peak = np.abs(x).max() if len(x) else 0
    buf = io.BytesIO()
    sf.write(buf, x * (0.9 / peak) if peak > 0 else x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _lut(name: str) -> list[list[int]]:
    from matplotlib import colormaps

    return (np.asarray(colormaps[name](np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(int).tolist()


def _clean(values, digits: int = 1) -> list:
    """JSON-friendly list: NaN -> null, rounded."""
    return [None if v is None or not np.isfinite(v) else round(float(v), digits) for v in values]


# ---------------------------------------------------------------------------
# page data
# ---------------------------------------------------------------------------

def praat_tracks(clip: Clip, hop: float = 0.01) -> dict:
    from .features import track

    tr = track(clip.audio, clip.sr, hop)
    voiced = tr.voiced                      # formants are only meaningful where voiced
    return {"t": _clean(tr.times, 3), "f0": _clean(tr.f0),
            "F": [_clean(np.where(voiced, tr.formants[:, i], np.nan), 0) for i in range(4)]}


def frames_overlay(frames: np.ndarray, times: np.ndarray) -> dict:
    """VocalTract frames (raw bytes, frame k at times[k]) -> Hz tracks and amplitude lanes."""
    frames = np.asarray(frames, dtype=np.float64)
    sounding = (frames[:, [GA_INDEX, AA_INDEX, FA_INDEX]] > 0).any(axis=1)
    gp_hz = 110.0 * 2.0 ** ((frames[:, GP_INDEX] - 100.0) / 48.0)
    return {
        "t": _clean(times, 4),
        "F": [_clean(frames[:, F1_INDEX + i] * FORMANT_SCALE, 0) for i in range(4)],
        "gp": _clean(np.where(frames[:, GA_INDEX] > 0, gp_hz, np.nan)),
        "ga": _clean(frames[:, GA_INDEX], 0), "aa": _clean(frames[:, AA_INDEX], 0),
        "fa": _clean(frames[:, FA_INDEX], 0), "on": sounding.astype(int).tolist(),
    }


def load_frames_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``<name>_frames.csv`` from resynth -> (frames (N, 13), times (N,))."""
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    return np.stack([data[n] for n in PARAM_NAMES], axis=1), data["time_s"]


def clip_payload(clip: Clip, fmax: float, *, praat: bool = True, words: list[dict] | None = None,
                 embed_audio: bool = True) -> dict:
    """Everything the page needs for one track."""
    return {
        "name": clip.name, "note": clip.note, "sr": clip.sr, "duration": clip.duration,
        "audio": base64.b64encode(wav_bytes(clip.audio, clip.sr)).decode("ascii") if embed_audio else None,
        "spec": clip_spectrograms(clip, fmax),
        "words": words or [],
        "praat": praat_tracks(clip) if praat else None,
    }


def _page_data(title: str, fmax: float, clips: list[dict], frames: dict | None,
               serve: dict | None = None) -> dict:
    return {
        "title": title, "fmax": fmax, "floorStored": FLOOR_DB, "diffRange": DIFF_RANGE_DB,
        "quietDb": DIFF_QUIET_DB, "lut": {"magma": _lut("magma"), "diverging": _lut("RdBu_r")},
        "clips": clips, "frames": frames, "serve": serve,
    }


def _render(data: dict) -> str:
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", _html_escape(data["title"])).replace("__DATA__", blob)


def build_page(clips: list[Clip], *, frames: np.ndarray | None = None,
               frame_times: np.ndarray | None = None, words: list[list[dict]] | None = None,
               praat: bool = True, fmax: float | None = None, title: str = "Spectrogram comparison") -> str:
    """Render a self-contained comparison page as an HTML string."""
    if not clips:
        raise ValueError("need at least one clip")
    fmax = float(fmax or min(c.sr for c in clips) / 2)
    payloads = [clip_payload(c, fmax, praat=praat,
                             words=words[i] if words and i < len(words) else None)
                for i, c in enumerate(clips)]
    overlay = frames_overlay(frames, frame_times) if frames is not None else None
    return _render(_page_data(title, fmax, payloads, overlay))


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def export_sv_layers(out_dir: Path, clips: list[Clip], frames: np.ndarray | None = None,
                     frame_times: np.ndarray | None = None,
                     words: list[list[dict]] | None = None) -> list[Path]:
    """CSV layers for Sonic Visualiser (File > Import Annotation Layer)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if frames is not None:
        ov = frames_overlay(frames, frame_times)
        on = np.asarray(ov["on"], dtype=bool)
        t = np.asarray(frame_times)
        series = {f"frames_f{i + 1}_hz": ov["F"][i] for i in range(4)}
        series["frames_pitch_hz"] = ov["gp"]
        for name, values in series.items():
            v = np.array([np.nan if x is None else x for x in values], dtype=float)
            keep = on & np.isfinite(v)
            p = out_dir / f"{name}.csv"
            np.savetxt(p, np.column_stack([t[keep], v[keep]]), delimiter=",", fmt="%.4f")
            paths.append(p)
    for i, c in enumerate(clips):
        if words and i < len(words) and words[i]:
            p = out_dir / f"words_{c.name}.csv"
            p.write_text("".join(f"{w['start']:.3f},{w['end'] - w['start']:.3f},"
                                 f"\"{w['word'].replace(chr(34), '')}\"\n" for w in words[i]),
                         encoding="utf-8")
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# folder browser (--serve)
# ---------------------------------------------------------------------------

def list_files(root: Path, limit: int = 5000) -> dict:
    """Audio files and ``*_frames.csv`` under ``root`` (relative POSIX paths, sorted)."""
    audio, frames = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS)
        for f in sorted(filenames):
            rel = (Path(dirpath) / f).relative_to(root).as_posix()
            if Path(f).suffix.lower() in AUDIO_EXTS:
                audio.append(rel)
            elif f.endswith("_frames.csv"):
                frames.append(rel)
        if len(audio) + len(frames) >= limit:
            break
    return {"audio": audio, "frames": frames}


class CompareServer:
    """Serves the comparison page for the files under ``root``, computing tracks on demand."""

    def __init__(self, root: Path, *, fmax: float = SERVE_FMAX, praat: bool = True,
                 asr_model: str = "small.en", title: str | None = None, cache_size: int = 64):
        self.root = root.resolve()
        self.fmax = float(fmax)
        self.praat = praat
        self.asr_model = asr_model
        self.title = title or f"Compare: {self.root.name or self.root}"
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def resolve(self, rel: str) -> Path:
        """A file under root; anything else (missing, outside root) is FileNotFoundError."""
        p = (self.root / rel).resolve()
        if not rel or not p.is_relative_to(self.root) or not p.is_file():
            raise FileNotFoundError(rel)
        return p

    def _cached(self, kind: str, rel: str, compute):
        p = self.resolve(rel)
        key = (kind, str(p), p.stat().st_mtime_ns)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        value = compute(p)
        with self._lock:
            self._cache[key] = value
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return value

    def _clip(self, p: Path) -> Clip:
        return load_clip(p, p.stem)

    def page(self, initial: list[str] = (), frames: str = "") -> str:
        serve = {"files": list_files(self.root), "initial": list(initial), "framesFile": frames,
                 "root": str(self.root)}
        return _render(_page_data(self.title, self.fmax, [], None, serve))

    def clip(self, rel: str) -> dict:
        return self._cached("clip", rel, lambda p: clip_payload(
            self._clip(p), self.fmax, praat=self.praat, embed_audio=False))

    def audio(self, rel: str) -> bytes:
        return self._cached("audio", rel, lambda p: wav_bytes(*(lambda c: (c.audio, c.sr))(self._clip(p))))

    def frames(self, rel: str) -> dict:
        return self._cached("frames", rel, lambda p: frames_overlay(*load_frames_csv(p)))

    def words(self, rel: str) -> dict:
        def compute(p: Path) -> dict:
            from .intelligibility import reference_for, transcribe_words, word_error_rate

            clip = self._clip(p)
            text, words = transcribe_words(clip.audio, clip.sr, self.asr_model)
            ref = reference_for(p)
            note = f"WER {word_error_rate(ref, text).wer:.0%}" if ref else ""
            return {"text": text, "words": words, "note": note}
        return self._cached("words", rel, compute)

    def make_handler(self, initial: list[str], frames: str):
        from http.server import BaseHTTPRequestHandler
        from urllib.parse import parse_qs, urlparse

        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlparse(self.path)
                rel = parse_qs(url.query).get("path", [""])[0]
                try:
                    if url.path in ("/", "/index.html"):
                        self._send(200, "text/html; charset=utf-8", app.page(initial, frames).encode())
                    elif url.path == "/api/files":
                        self._json(list_files(app.root))
                    elif url.path == "/api/clip":
                        self._json(app.clip(rel))
                    elif url.path == "/api/audio":
                        self._send(200, "audio/wav", app.audio(rel))
                    elif url.path == "/api/frames":
                        self._json(app.frames(rel))
                    elif url.path == "/api/words":
                        self._json(app.words(rel))
                    else:
                        self._json({"error": "not found"}, 404)
                except FileNotFoundError:
                    self._json({"error": f"no such file under {app.root}: {rel}"}, 404)
                except Exception as e:      # report to the page instead of dropping the connection
                    self.log_error("%s %s: %r", url.path, rel, e)
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)

            def _send(self, code: int, ctype: str, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200):
                self._send(code, "application/json", json.dumps(obj, separators=(",", ":")).encode())

            def log_message(self, fmt, *args):     # quiet; errors still go through log_error
                pass

            def log_error(self, fmt, *args):
                print("compare server:", fmt % args, flush=True)

        return Handler


def serve(root: Path, initial: list[Path] = (), *, frames: Path | None = None, port: int = 8765,
          open_browser: bool = True, **kwargs) -> None:
    """Run the folder browser on http://127.0.0.1:port/ until interrupted."""
    from http.server import ThreadingHTTPServer

    app = CompareServer(root, **kwargs)

    def rel(p: Path) -> str:
        return p.resolve().relative_to(app.root).as_posix()

    handler = app.make_handler([rel(p) for p in initial], rel(frames) if frames else "")
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:                          # port taken: let the OS pick one
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    n = len(list_files(app.root)["audio"])
    print(f"Comparing {n} audio files under {app.root}\n  open {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Side-by-side spectrogram comparison page")
    ap.add_argument("inputs", nargs="*", type=Path,
                    help="audio files, top to bottom (with --serve: the tracks to start with)")
    ap.add_argument("--serve", type=Path, nargs="?", const=Path("."), metavar="DIR",
                    help="run a local folder browser over DIR (default .) instead of writing a page")
    ap.add_argument("--port", type=int, default=8765, help="--serve port (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="--serve: don't open a browser")
    ap.add_argument("-o", "--out", type=Path, default=Path("out") / "compare.html")
    ap.add_argument("--names", help="comma-separated display names")
    ap.add_argument("--frames", type=Path, help="resynth <name>_frames.csv to overlay")
    ap.add_argument("--words", action="store_true", help="Whisper word timings per file")
    ap.add_argument("--asr", default="small.en", help="Whisper model for --words")
    ap.add_argument("--no-praat", action="store_true", help="skip Praat pitch/formant tracks")
    ap.add_argument("--start", type=float, default=0.0, help="trim: start time (s), all files")
    ap.add_argument("--duration", type=float, help="trim: length (s)")
    ap.add_argument("--fmax", type=float,
                    help=f"top frequency (default: lowest Nyquist; --serve: {SERVE_FMAX:.0f})")
    ap.add_argument("--title", help="page title")
    ap.add_argument("--sv-dir", type=Path, help="also export overlays as Sonic Visualiser CSV layers")
    args = ap.parse_args(argv)

    if args.serve is not None:
        serve(args.serve, args.inputs, frames=args.frames, port=args.port,
              open_browser=not args.no_browser, fmax=args.fmax or SERVE_FMAX,
              praat=not args.no_praat, asr_model=args.asr, title=args.title)
        return
    if not args.inputs:
        ap.error("give audio files, or --serve DIR")

    names = args.names.split(",") if args.names else [None] * len(args.inputs)
    clips = [load_clip(p, n, args.start, args.duration) for p, n in zip(args.inputs, names)]

    frames = times = None
    if args.frames:
        frames, times = load_frames_csv(args.frames)
        times = times - args.start

    words = None
    if args.words:
        from .intelligibility import transcribe_words
        words = [transcribe_words(c.audio, c.sr, args.asr)[1] for c in clips]

    title = args.title or " vs ".join(c.name for c in clips)
    html = build_page(clips, frames=frames, frame_times=times, words=words,
                      praat=not args.no_praat, fmax=args.fmax, title=title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"{args.out} ({len(html) / 1e6:.1f} MB)")
    if args.sv_dir:
        for p in export_sv_layers(args.sv_dir, clips, frames, times, words):
            print(f"  layer -> {p}")


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f5f5f3;--panel:#ffffff;--text:#1c1c1e;--muted:#6a6a70;--border:#d8d8dd;--accent:#2f6fde;--axis:#5a5a60;--grid:rgba(0,0,0,.08);--empty:#e9e9ec;--err:#c0392b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121214;--panel:#1b1b1e;--text:#ececef;--muted:#9898a0;--border:#323237;--accent:#6aa0ff;--axis:#a2a2aa;--grid:rgba(255,255,255,.08);--empty:#232327;--err:#ff7b6b}}
:root[data-theme="dark"]{--bg:#121214;--panel:#1b1b1e;--text:#ececef;--muted:#9898a0;--border:#323237;--accent:#6aa0ff;--axis:#a2a2aa;--grid:rgba(255,255,255,.08);--empty:#232327;--err:#ff7b6b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--border);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center}
header h1{font-size:15px;margin:0 8px 0 0;font-weight:650}
.group{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
label{color:var(--muted);font-size:13px;display:flex;gap:5px;align-items:center}
select,input[type=range]{accent-color:var(--accent)}
select{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:3px 6px;font:inherit;font-size:13px;max-width:100%}
button{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 10px;font:inherit;font-size:13px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
button.active{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
button.icon{padding:2px 8px;color:var(--muted)}
button .k{color:var(--muted);font-size:11px;margin-left:4px}
main{padding:4px 0 8px}
.pane{margin:8px 16px;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.pane .head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:6px 10px 4px;font-size:13px}
.pane .head b{font-weight:650}
.pane .head select.file{min-width:0;flex:0 1 420px}
.pane .head .note{color:var(--muted)}
.pane .head .note.err{color:var(--err)}
.pane .head .legend{margin-left:auto;color:var(--muted);font-size:12px;display:flex;gap:10px;flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px;box-shadow:0 0 0 1px var(--border)}
canvas{display:block;width:100%;touch-action:none;cursor:grab}
canvas.dragging{cursor:grabbing}
.help{color:var(--muted);font-size:12px;margin:4px 16px 18px}
.hidden{display:none !important}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="group" id="players"></div>
  <div class="group">
    <label>View <select id="res"><option value="wide">Formants (5 ms window)</option><option value="narrow">Harmonics (35 ms window)</option></select></label>
    <label>Floor <input type="range" id="floor" min="-100" max="-30" step="5" value="-70"><span id="floorv">-70 dB</span></label>
    <label>Max <select id="fmax"></select></label>
  </div>
  <div class="group">
    <label id="lblFrames"><input type="checkbox" id="ovFrames" checked> frames</label>
    <label id="lblPraat"><input type="checkbox" id="ovPraat"> Praat formants</label>
    <label id="lblWords"><input type="checkbox" id="ovWords" checked> words</label>
    <label id="lblDiff"><input type="checkbox" id="ovDiff"> difference</label>
  </div>
  <div class="group" id="serveControls">
    <button id="addTrack" title="add a track">+ track</button>
    <label>Frames file <select id="framesFile"></select></label>
    <button id="rescan" class="icon" title="rescan the folder for new files">↻ rescan</button>
  </div>
</header>
<main id="panes"></main>
<p class="help"><span id="serveHelp">Pick a file in each track's menu &middot; </span>URL options: <code>#t0=2&amp;t1=4&amp;res=narrow&amp;floor=-60&amp;fmax=5000&amp;diff=1&amp;praat=1&amp;words=0</code> &middot; Wheel or pinch: zoom &middot; drag: pan &middot; double-click: whole file &middot; click: move playhead &middot; <b>space</b>: play/pause &middot; <b>1&ndash;9</b>: switch track at the same moment (A/B) &middot; <b>&larr;/&rarr;</b>: step 0.5 s</p>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const SERVE = !!D.serve;
const RES = ["wide", "narrow"];
const L = 46, R = 10;                      // plot margins (css px)
const SPEC_H = 220, WORDS_H = 22, AXIS_H = 18, LANE_H = 110;
const CLIP_COLORS = ["#ffb000", "#35d0ff", "#ff5fa2", "#8ee05a", "#c08bff", "#ff7b39"];
const state = {t0: 0, t1: 1, res: "wide", floor: -70, fmax: D.fmax, active: 0, cursor: 0,
               ovFrames: true, ovPraat: false, ovWords: true, ovDiff: false};
const $ = id => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
document.title = D.title; $("title").textContent = D.title;

// A track: {path, name, note, sr, duration, audio (Audio), gray {res: {w,h,g,hop}}, colored {res: canvas},
//           praat, words, loading, error}. A track with no file loaded has no `gray`.
let tracks = [], frames = D.frames, framesPath = "", files = SERVE ? D.serve.files : {audio: [], frames: []};
let duration = 1, diffImg = {}, pitchRange = null, dirty = true;
const loaded = t => !!(t && t.gray);
const basename = p => p.split("/").pop().replace(/\.[^.]+$/, "");

// ---------- data ----------
function decodeGray(b64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const cv = el("canvas", {width: img.width, height: img.height});
      const cx = cv.getContext("2d"); cx.drawImage(img, 0, 0);
      const px = cx.getImageData(0, 0, img.width, img.height).data;
      const g = new Uint8Array(img.width * img.height);
      for (let i = 0; i < g.length; i++) g[i] = px[i * 4];
      resolve({w: img.width, h: img.height, g});
    };
    img.onerror = reject;
    img.src = "data:image/png;base64," + b64;
  });
}
function colorize(G, lut, map) {
  const cv = el("canvas", {width: G.w, height: G.h});
  const cx = cv.getContext("2d"), im = cx.createImageData(G.w, G.h), d = im.data;
  const table = new Array(256); for (let v = 0; v < 256; v++) table[v] = lut[map(v)];
  for (let i = 0, j = 0; i < G.g.length; i++, j += 4) { const c = table[G.g[i]]; d[j] = c[0]; d[j + 1] = c[1]; d[j + 2] = c[2]; d[j + 3] = 255; }
  cx.putImageData(im, 0, 0); return cv;
}
function specMap(v) {
  const db = D.floorStored + v / 255 * -D.floorStored;
  return Math.max(0, Math.min(255, Math.round((db - state.floor) / -state.floor * 255)));
}
function colorTrack(t) { t.colored = {}; for (const r in t.gray) t.colored[r] = colorize(t.gray[r], D.lut.magma, specMap); }
function recolor() { tracks.forEach(t => loaded(t) && colorTrack(t)); dirty = true; }

async function fillTrack(t, c, audioSrc) {
  Object.assign(t, {name: c.name, note: c.note || "", sr: c.sr, duration: c.duration, praat: c.praat,
                    words: c.words || [], loading: false, error: ""});
  const gray = {};
  for (const r of RES) gray[r] = Object.assign(await decodeGray(c.spec[r].png), {hop: c.spec[r].hop});
  t.gray = gray; colorTrack(t);
  t.audio = new Audio(audioSrc); t.audio.preload = "auto";
  t.audio.addEventListener("ended", () => { if (tracks[state.active] === t) { state.cursor = t.duration; updateButtons(); dirty = true; } });
}
async function api(path, rel) {
  const r = await fetch(`${path}?path=${encodeURIComponent(rel)}`);
  const body = await r.json().catch(() => ({error: `HTTP ${r.status}`}));
  if (!r.ok || body.error) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}
async function loadTrack(i, rel) {
  const old = tracks[i]; if (old && old.audio) old.audio.pause();
  const t = {path: rel, name: rel ? basename(rel) : "", loading: !!rel, error: "", words: []};
  tracks[i] = t; changed();
  if (!rel) return;
  try { await fillTrack(t, await api("api/clip", rel), `api/audio?path=${encodeURIComponent(rel)}`); }
  catch (e) { t.loading = false; t.error = e.message; }
  if (tracks[i] === t) changed();
}
async function loadWords(t, btn) {
  btn.disabled = true; btn.textContent = "transcribing…";
  try { const w = await api("api/words", t.path); t.words = w.words; t.note = w.note || t.note; state.ovWords = true; $("ovWords").checked = true; }
  catch (e) { t.error = e.message; }
  changed();
}
async function loadFrames(rel) {
  framesPath = rel; frames = null;
  if (rel) { try { frames = await api("api/frames", rel); } catch (e) { alert(`frames: ${e.message}`); framesPath = ""; } }
  changed();
}

// The difference pane: track 2 minus track 1 in dB, from the 8-bit spectrograms (any time grids).
function computeDiff() {
  diffImg = {};
  const [a, b] = tracks; if (!loaded(a) || !loaded(b)) return;
  const unit = -D.floorStored / 255, quiet = (D.quietDb - D.floorStored) / unit;
  for (const r of RES) {
    const A = a.gray[r], B = b.gray[r]; if (A.h !== B.h) continue;
    const hop = Math.min(A.hop, B.hop), W = Math.ceil(duration / hop) + 1, H = A.h, g = new Uint8Array(W * H);
    for (let x = 0; x < W; x++) {
      const ca = Math.round(x * hop / A.hop), cb = Math.round(x * hop / B.hop);
      for (let y = 0; y < H; y++) {
        const va = ca < A.w ? A.g[y * A.w + ca] : 0, vb = cb < B.w ? B.g[y * B.w + cb] : 0;
        const d = (va < quiet && vb < quiet) ? 0 : (vb - va) * unit;
        g[y * W + x] = Math.round(128 + Math.max(-1, Math.min(1, d / D.diffRange)) * 127);
      }
    }
    diffImg[r] = {canvas: colorize({w: W, h: H, g}, D.lut.diverging, v => v), hop};
  }
}

// Call after any change to tracks or frames: recomputes extents and rebuilds the panes.
function changed() {
  const durs = tracks.filter(loaded).map(t => t.duration);
  const newDur = Math.max(0.1, ...durs);
  if (newDur !== duration) { const whole = state.t0 <= 0 && state.t1 >= duration - 1e-6; duration = newDur; if (whole || state.t1 > duration) setView(whole ? 0 : state.t0, whole ? duration : state.t1); }
  if (state.active >= tracks.length || !loaded(tracks[state.active])) { const k = tracks.findIndex(loaded); state.active = Math.max(0, k); }
  state.cursor = Math.min(state.cursor, duration);
  computeDiff(); pitchRange = null; build(); writeHash();
}

// ---------- audio ----------
const activeAudio = () => (tracks[state.active] && tracks[state.active].audio) || null;
const playing = () => { const a = activeAudio(); return !!a && !a.paused; };
function nowT() { return playing() ? activeAudio().currentTime : state.cursor; }
function switchTo(i, play) {
  if (!loaded(tracks[i])) return;
  const t = nowT(), wasPlaying = playing();
  if (activeAudio()) activeAudio().pause();
  state.active = i; state.cursor = Math.min(t, tracks[i].duration);
  tracks[i].audio.currentTime = state.cursor;
  if (play ?? wasPlaying) tracks[i].audio.play();
  updateButtons(); dirty = true;
}
function togglePlay() {
  const a = activeAudio(); if (!a) return;
  if (a.paused) { if (state.cursor >= tracks[state.active].duration - 0.01) state.cursor = 0; a.currentTime = state.cursor; a.play(); }
  else { a.pause(); state.cursor = a.currentTime; }
  updateButtons();
}
function seek(t) {
  const a = activeAudio(); if (!a) return;
  state.cursor = Math.max(0, Math.min(t, tracks[state.active].duration));
  a.currentTime = state.cursor; dirty = true;
}
let btns = [];
function buildButtons() {
  $("players").replaceChildren(); btns = [];
  tracks.forEach((t, i) => {
    if (!loaded(t)) { btns.push(null); return; }
    const b = el("button");
    b.innerHTML = `<span class="sw" style="background:${CLIP_COLORS[i % 6]}"></span><span class="lbl"></span><span class="k">${i + 1}</span>`;
    b.onclick = () => { if (state.active === i) togglePlay(); else switchTo(i, true); };
    $("players").appendChild(b); btns.push(b);
  });
  updateButtons();
}
function updateButtons() {
  btns.forEach((b, i) => {
    if (!b) return;
    const on = i === state.active;
    b.classList.toggle("active", on);
    b.querySelector(".lbl").textContent = (on && playing() ? "❚❚ " : "▶ ") + tracks[i].name;
  });
}

// ---------- panes ----------
let panes = [];
const sw = (c, t) => `<span><span class="sw" style="background:${c}"></span>${t}</span>`;
function makePane(kind, idx, head) {
  const div = el("div", {className: "pane"}), cv = el("canvas");
  div.append(el("div", {className: "head"}, ...head), cv); $("panes").appendChild(div);
  const p = {kind, idx, div, cv, ctx: cv.getContext("2d")};
  attachInput(p); panes.push(p); return p;
}
function legend(html) { const s = el("span", {className: "legend"}); s.innerHTML = html; return s; }
function fileSelect(current) {
  const sel = el("select", {className: "file", title: "choose the audio file for this track"});
  sel.add(new Option("— choose a file —", ""));
  const groups = new Map();
  for (const f of files.audio) { const dir = f.includes("/") ? f.slice(0, f.lastIndexOf("/")) : "."; if (!groups.has(dir)) groups.set(dir, []); groups.get(dir).push(f); }
  for (const [dir, fs] of groups) {
    const og = el("optgroup", {label: dir === "." ? "(top folder)" : dir + "/"});
    for (const f of fs) og.appendChild(new Option(f.split("/").pop(), f));
    sel.appendChild(og);
  }
  if (current && !files.audio.includes(current)) sel.add(new Option(current + " (missing)", current));
  sel.value = current || "";
  return sel;
}
function trackHead(i) {
  const t = tracks[i], kids = [];
  const num = el("b", {textContent: SERVE ? `${i + 1}.` : `${i + 1}. ${t.name}`});
  num.style.color = CLIP_COLORS[i % 6]; kids.push(num);
  if (SERVE) {
    const sel = fileSelect(t.path); sel.onchange = () => loadTrack(i, sel.value); kids.push(sel);
  }
  const note = el("span", {className: "note" + (t.error ? " err" : "")});
  note.textContent = t.error || (t.loading ? "loading…" : [t.note, loaded(t) ? `${t.duration.toFixed(2)} s · ${t.sr / 1000} kHz` : ""].filter(Boolean).join(" · "));
  kids.push(note);
  if (SERVE && loaded(t) && !t.words.length) {
    const wb = el("button", {className: "icon", textContent: "words", title: "Whisper word timings (and WER if a .txt transcript is next to the file)"});
    wb.onclick = () => loadWords(t, wb); kids.push(wb);
  }
  if (SERVE && tracks.length > 1) {
    const x = el("button", {className: "icon", textContent: "✕", title: "remove this track"});
    x.onclick = () => { if (t.audio) t.audio.pause(); tracks.splice(i, 1); changed(); }; kids.push(x);
  }
  kids.push(legend((frames ? sw("#35e0ff", "frames formants") : "") + (tracks.some(t => t.praat) ? sw("#ffffff", "Praat formants") : "")));
  return kids;
}
let diffPane = null;
function build() {
  $("panes").replaceChildren(); panes = [];
  tracks.forEach((t, i) => makePane("spec", i, trackHead(i)));
  const [a, b] = tracks;
  diffPane = loaded(a) && loaded(b) ? makePane("diff", 0, [el("b", {textContent: `difference: ${b.name} − ${a.name}`}),
    legend(sw("#c0392b", `${b.name} louder`) + sw("#2e6db4", `${a.name} louder`) + `<span>±${D.diffRange} dB</span>`)]) : null;
  const hasPraat = tracks.some(t => loaded(t) && t.praat);
  if (hasPraat || frames) makePane("pitch", 0, [el("b", {textContent: "pitch"}),
    legend(tracks.map((t, i) => loaded(t) && t.praat ? sw(CLIP_COLORS[i % 6], `Praat f0: ${t.name}`) : "").join("") + (frames ? sw("var(--text)", "frames gp") : ""))]);
  if (frames) makePane("amp", 0, [el("b", {textContent: "source amplitudes (frames)"}),
    legend(sw("#ff9f1c", "ga voice") + sw("#2ec4b6", "aa aspiration") + sw("#b388ff", "fa frication"))]);
  $("lblFrames").classList.toggle("hidden", !frames);
  $("lblPraat").classList.toggle("hidden", !hasPraat);
  $("lblWords").classList.toggle("hidden", !tracks.some(t => t.words && t.words.length));
  $("lblDiff").classList.toggle("hidden", !diffPane);
  if (diffPane) diffPane.div.classList.toggle("hidden", !state.ovDiff);
  buildButtons(); resize();
}
function paneHeight(p) {
  if (p.kind === "spec") return SPEC_H + (state.ovWords && tracks[p.idx].words.length ? WORDS_H : 0) + AXIS_H;
  if (p.kind === "diff") return SPEC_H + AXIS_H;
  return LANE_H + AXIS_H;
}
function resize() {
  const dpr = window.devicePixelRatio || 1;
  for (const p of panes) {
    const w = p.cv.clientWidth, h = paneHeight(p);
    p.cv.style.height = h + "px"; p.cv.width = Math.round(w * dpr); p.cv.height = Math.round(h * dpr);
    p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); p.w = w; p.h = h;
  }
  dirty = true;
}

// ---------- drawing ----------
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function tx(p, t) { return L + (t - state.t0) / (state.t1 - state.t0) * (p.w - L - R); }
function xt(p, x) { return state.t0 + (x - L) / (p.w - L - R) * (state.t1 - state.t0); }
function niceStep(span, target) {
  const raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * mag >= raw) return m * mag;
  return 10 * mag;
}
function drawTimeAxis(p, y) {
  const c = p.ctx, step = niceStep(state.t1 - state.t0, Math.max(2, (p.w - L) / 90));
  c.fillStyle = css("--axis"); c.strokeStyle = css("--axis"); c.font = "11px system-ui"; c.textAlign = "center";
  for (let t = Math.ceil(state.t0 / step) * step; t <= state.t1 + 1e-9; t += step) {
    const x = tx(p, t); c.beginPath(); c.moveTo(x, y); c.lineTo(x, y + 4); c.stroke();
    c.fillText(t.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0) + " s", x, y + 14);
  }
}
// Image column k is centred at k * hop; only the part of the view the image covers is drawn.
function drawImageView(p, img, hop, top, H) {
  const c = p.ctx, rows = img.height, shown = Math.max(2, Math.round(rows * state.fmax / D.fmax));
  const s0 = state.t0 / hop, s1 = state.t1 / hop, c0 = Math.max(0, s0), c1 = Math.min(img.width, s1), W = p.w - L - R;
  if (c1 <= c0) return;
  const x0 = L + (c0 - s0) / (s1 - s0) * W, x1 = L + (c1 - s0) / (s1 - s0) * W;
  c.imageSmoothingEnabled = true;
  c.drawImage(img, c0, rows - shown, c1 - c0, shown, x0, top, x1 - x0, H);
}
function drawFreqAxis(p, top, H) {
  const c = p.ctx, step = niceStep(state.fmax, 5);
  c.fillStyle = css("--axis"); c.font = "11px system-ui"; c.textAlign = "right"; c.textBaseline = "middle";
  for (let f = 0; f <= state.fmax + 1e-6; f += step) {
    const y = top + H * (1 - f / state.fmax);
    c.fillText(f >= 1000 ? (f / 1000) + "k" : f, L - 6, Math.min(Math.max(y, top + 6), top + H - 6));
  }
  c.textBaseline = "alphabetic";
}
function dots(p, ts, vs, on, top, H, color, r, scaleY) {
  const c = p.ctx; c.fillStyle = color;
  for (let k = 0; k < ts.length; k++) {
    const t = ts[k], v = vs[k];
    if (v === null || t < state.t0 || t > state.t1 || (on && !on[k])) continue;
    const y = scaleY(v); if (y < top || y > top + H) continue;
    c.fillRect(tx(p, t) - r, y - r, 2 * r, 2 * r);
  }
}
function line(p, ts, vs, top, H, color, scaleY, width) {
  const c = p.ctx; c.save(); c.beginPath(); c.rect(L, top, p.w - L - R, H); c.clip();
  c.strokeStyle = color; c.lineWidth = width || 1.4; c.beginPath(); let pen = false;
  for (let k = 0; k < ts.length; k++) {
    const v = vs[k];
    if (v === null) { pen = false; continue; }
    const x = tx(p, ts[k]), y = scaleY(v);
    if (pen) c.lineTo(x, y); else c.moveTo(x, y); pen = true;
  }
  c.stroke(); c.restore();
}
function drawPlayhead(p, top, H) {
  const t = nowT(); if (!activeAudio() || t < state.t0 || t > state.t1) return;
  const c = p.ctx, x = tx(p, t);
  c.strokeStyle = "rgba(255,255,255,.9)"; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, top); c.lineTo(x, top + H); c.stroke();
  c.strokeStyle = "rgba(0,0,0,.5)"; c.lineWidth = 1; c.beginPath(); c.moveTo(x + 1.5, top); c.lineTo(x + 1.5, top + H); c.stroke();
}
function drawPlaceholder(p, text) {
  const c = p.ctx; c.fillStyle = css("--empty"); c.fillRect(L, 0, p.w - L - R, SPEC_H);
  c.fillStyle = css("--muted"); c.font = "13px system-ui"; c.textAlign = "center"; c.textBaseline = "middle";
  c.fillText(text, L + (p.w - L - R) / 2, SPEC_H / 2); c.textBaseline = "alphabetic";
}
function drawSpec(p) {
  const c = p.ctx, t = tracks[p.idx], top = 0, H = SPEC_H;
  c.clearRect(0, 0, p.w, p.h);
  if (!loaded(t)) { drawPlaceholder(p, t.loading ? "computing spectrogram…" : t.error ? "could not load this file" : "choose a file above"); drawTimeAxis(p, H); return; }
  drawImageView(p, t.colored[state.res], t.gray[state.res].hop, top, H); drawFreqAxis(p, top, H);
  const fy = f => top + H * (1 - f / state.fmax);
  if (state.ovPraat && t.praat) for (const F of t.praat.F) dots(p, t.praat.t, F, null, top, H, "rgba(255,255,255,.75)", 1.2, fy);
  if (state.ovFrames && frames) for (const F of frames.F) dots(p, frames.t, F, frames.on, top, H, "rgba(53,224,255,.85)", 1.4, fy);
  if (p.idx === state.active) { c.strokeStyle = CLIP_COLORS[p.idx % 6]; c.lineWidth = 3; c.beginPath(); c.moveTo(L - 2, top); c.lineTo(L - 2, top + H); c.stroke(); c.lineWidth = 1; }
  let y = top + H;
  if (state.ovWords && t.words.length) { drawWords(p, t.words, y, WORDS_H); y += WORDS_H; }
  drawPlayhead(p, top, y - top); drawTimeAxis(p, y);
}
function drawWords(p, words, top, H) {
  const c = p.ctx; c.font = "12px system-ui"; c.textBaseline = "middle"; c.textAlign = "left";
  c.fillStyle = css("--grid"); c.fillRect(L, top, p.w - L - R, H);
  for (const w of words) {
    if (w.end < state.t0 || w.start > state.t1) continue;
    const x0 = Math.max(tx(p, w.start), L), x1 = Math.min(tx(p, w.end), p.w - R);
    c.strokeStyle = css("--axis"); c.beginPath(); c.moveTo(tx(p, w.start), top + 3); c.lineTo(tx(p, w.start), top + H - 3); c.stroke();
    c.save(); c.beginPath(); c.rect(x0, top, Math.max(0, x1 - x0), H); c.clip();
    c.fillStyle = css("--text"); c.fillText(w.word, x0 + 3, top + H / 2); c.restore();
  }
  c.textBaseline = "alphabetic";
}
function drawDiff(p) {
  const c = p.ctx; c.clearRect(0, 0, p.w, p.h);
  const d = diffImg[state.res]; if (!d) return;
  drawImageView(p, d.canvas, d.hop, 0, SPEC_H); drawFreqAxis(p, 0, SPEC_H);
  drawPlayhead(p, 0, SPEC_H); drawTimeAxis(p, SPEC_H);
}
function drawPitch(p) {
  const c = p.ctx, top = 4, H = LANE_H - 8; c.clearRect(0, 0, p.w, p.h);
  if (!pitchRange) {
    const all = [];
    tracks.forEach(t => loaded(t) && t.praat && t.praat.f0.forEach(v => v && all.push(v)));
    if (frames) frames.gp.forEach(v => v && all.push(v));
    all.sort((a, b) => a - b);
    const lo = all.length ? all[Math.floor(all.length * .02)] : 80, hi = all.length ? all[Math.floor(all.length * .98)] : 300;
    pitchRange = [Math.max(40, lo / 1.25), Math.min(1000, hi * 1.25)];
  }
  const [lo, hi] = pitchRange, py = f => top + H * (1 - Math.log(f / lo) / Math.log(hi / lo));
  c.fillStyle = css("--grid"); c.fillRect(L, top, p.w - L - R, H);
  c.fillStyle = css("--axis"); c.font = "11px system-ui"; c.textAlign = "right"; c.textBaseline = "middle";
  for (const f of [50, 75, 100, 150, 200, 300, 400, 600, 800]) if (f >= lo && f <= hi) { c.fillText(f, L - 6, py(f)); }
  c.textBaseline = "alphabetic";
  tracks.forEach((t, i) => { if (loaded(t) && t.praat) dots(p, t.praat.t, t.praat.f0, null, top, H, CLIP_COLORS[i % 6], 1.6, py); });
  if (frames) line(p, frames.t, frames.gp, top, H, css("--text"), py, 1.2);
  drawPlayhead(p, top, H); drawTimeAxis(p, LANE_H);
}
function drawAmp(p) {
  const c = p.ctx, top = 4, H = LANE_H - 8; c.clearRect(0, 0, p.w, p.h);
  c.fillStyle = css("--grid"); c.fillRect(L, top, p.w - L - R, H);
  const ay = v => top + H * (1 - v / 255);
  c.fillStyle = css("--axis"); c.font = "11px system-ui"; c.textAlign = "right"; c.textBaseline = "middle";
  for (const v of [0, 128, 255]) c.fillText(v, L - 6, ay(v));
  c.textBaseline = "alphabetic";
  line(p, frames.t, frames.ga, top, H, "#ff9f1c", ay);
  line(p, frames.t, frames.aa, top, H, "#2ec4b6", ay);
  line(p, frames.t, frames.fa, top, H, "#b388ff", ay);
  drawPlayhead(p, top, H); drawTimeAxis(p, LANE_H);
}
function drawAll() {
  for (const p of panes) {
    if (p.div.classList.contains("hidden")) continue;
    if (p.kind === "spec") drawSpec(p); else if (p.kind === "diff") drawDiff(p);
    else if (p.kind === "pitch") drawPitch(p); else drawAmp(p);
  }
}
function loop() {
  if (playing()) {
    const t = activeAudio().currentTime, span = state.t1 - state.t0;
    if (t > state.t1 || t < state.t0) { state.t0 = Math.max(0, Math.min(t - span * 0.1, duration - span)); state.t1 = state.t0 + span; }
    dirty = true;
  }
  if (dirty) { dirty = false; drawAll(); }
  requestAnimationFrame(loop);
}

// ---------- interaction ----------
function setView(t0, t1) {
  const span = Math.max(0.05, Math.min(duration, t1 - t0));
  t0 = Math.max(0, Math.min(t0, duration - span));
  state.t0 = t0; state.t1 = t0 + span; dirty = true;
}
function attachInput(p) {
  let drag = null;
  p.cv.addEventListener("pointerdown", e => { drag = {x: e.offsetX, t0: state.t0, t1: state.t1, moved: false}; p.cv.setPointerCapture(e.pointerId); });
  p.cv.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.offsetX - drag.x; if (Math.abs(dx) > 3) { drag.moved = true; p.cv.classList.add("dragging"); }
    if (drag.moved) { const dt = -dx / (p.w - L - R) * (drag.t1 - drag.t0); setView(drag.t0 + dt, drag.t1 + dt); }
  });
  p.cv.addEventListener("pointerup", e => {
    if (drag && !drag.moved && e.offsetX >= L) {
      if (p.kind === "spec" && p.idx !== state.active) switchTo(p.idx);
      seek(xt(p, e.offsetX));
    }
    drag = null; p.cv.classList.remove("dragging");
  });
  p.cv.addEventListener("wheel", e => {
    e.preventDefault();
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) { const dt = e.deltaX / (p.w - L - R) * (state.t1 - state.t0); setView(state.t0 + dt, state.t1 + dt); return; }
    const tm = xt(p, e.offsetX), k = Math.exp(e.deltaY * 0.0015);
    setView(tm - (tm - state.t0) * k, tm + (state.t1 - tm) * k);
  }, {passive: false});
  p.cv.addEventListener("dblclick", () => setView(0, duration));
}
window.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (/^Digit[1-9]$/.test(e.code)) { const i = +e.code.slice(5) - 1; if (i < tracks.length) switchTo(i); }
  else if (e.code === "ArrowRight") seek(nowT() + 0.5);
  else if (e.code === "ArrowLeft") seek(nowT() - 0.5);
  else if (e.key === "+" || e.key === "=") { const m = (state.t0 + state.t1) / 2, h = (state.t1 - state.t0) / 2.5; setView(m - h, m + h); }
  else if (e.key === "-") { const m = (state.t0 + state.t1) / 2, h = (state.t1 - state.t0); setView(m - h, m + h); }
});
$("res").onchange = e => { state.res = e.target.value; dirty = true; };
$("floor").oninput = e => { state.floor = +e.target.value; $("floorv").textContent = state.floor + " dB"; recolor(); };
for (const f of [2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000]) if (f < D.fmax) $("fmax").add(new Option((f / 1000) + " kHz", f));
$("fmax").add(new Option((D.fmax / 1000).toFixed(1).replace(".0", "") + " kHz", D.fmax)); $("fmax").value = D.fmax;
$("fmax").onchange = e => { state.fmax = +e.target.value; dirty = true; };
for (const id of ["ovFrames", "ovPraat", "ovWords"]) $(id).onchange = e => { state[id] = e.target.checked; resize(); };
$("ovDiff").onchange = e => { state.ovDiff = e.target.checked; if (diffPane) diffPane.div.classList.toggle("hidden", !state.ovDiff); resize(); };
window.addEventListener("resize", resize);

// ---------- folder browser controls ----------
function fillFramesSelect() {
  const sel = $("framesFile"); sel.replaceChildren(new Option("none", ""));
  for (const f of files.frames) sel.add(new Option(f, f));
  if (framesPath && !files.frames.includes(framesPath)) sel.add(new Option(framesPath + " (missing)", framesPath));
  sel.value = framesPath;
}
if (SERVE) {
  $("addTrack").onclick = () => { tracks.push({path: "", name: "", words: []}); changed(); };
  $("framesFile").onchange = e => loadFrames(e.target.value);
  $("rescan").onclick = async () => {
    try { const r = await fetch("api/files"); files = await r.json(); } catch (e) { alert(`rescan: ${e.message}`); return; }
    fillFramesSelect(); build();
  };
} else {
  $("serveControls").classList.add("hidden"); $("serveHelp").classList.add("hidden");
}

// ---------- URL options (#t0=..&t1=..&res=narrow&floor=-60&fmax=5000&diff=1&praat=1&frames=0&words=0
//            folder browser: &files=a.wav,b.wav&framesfile=x_frames.csv) ----------
const hashIn = new URLSearchParams(location.hash.slice(1));
function applyHash(h) {
  if (h.has("res") && RES.includes(h.get("res"))) { state.res = h.get("res"); $("res").value = state.res; }
  if (h.has("floor")) { state.floor = Math.max(-100, Math.min(-10, +h.get("floor"))); $("floor").value = state.floor; $("floorv").textContent = state.floor + " dB"; }
  if (h.has("fmax")) { state.fmax = Math.min(D.fmax, +h.get("fmax")); if (![...$("fmax").options].some(o => +o.value === state.fmax)) $("fmax").add(new Option(state.fmax + " Hz", state.fmax)); $("fmax").value = state.fmax; }
  for (const [key, id] of [["frames", "ovFrames"], ["praat", "ovPraat"], ["words", "ovWords"], ["diff", "ovDiff"]])
    if (h.has(key)) { state[id] = h.get(key) === "1"; $(id).checked = state[id]; }
}
function writeHash() {
  if (!SERVE) return;
  const h = new URLSearchParams(location.hash.slice(1));
  h.set("files", tracks.map(t => t.path).join(","));
  if (framesPath) h.set("framesfile", framesPath); else h.delete("framesfile");
  for (const k of ["t0", "t1"]) h.delete(k);      // the view is only applied once, at load
  history.replaceState(null, "", "#" + h.toString().replace(/%2F/g, "/").replace(/%2C/g, ","));
}

// ---------- start ----------
(async () => {
  applyHash(hashIn);
  if (SERVE) {
    const initial = hashIn.has("files") ? hashIn.get("files").split(",") : D.serve.initial;
    framesPath = hashIn.has("framesfile") ? hashIn.get("framesfile") : D.serve.framesFile;
    fillFramesSelect();
    tracks = (initial.length ? initial : ["", ""]).map(p => ({path: p, name: "", words: []}));
    changed(); requestAnimationFrame(loop);
    await Promise.all([...tracks.map((t, i) => t.path ? loadTrack(i, t.path) : null), framesPath ? loadFrames(framesPath) : null]);
  } else {
    tracks = D.clips.map(() => ({path: "", words: []}));
    for (let i = 0; i < D.clips.length; i++) await fillTrack(tracks[i], D.clips[i], "data:audio/wav;base64," + D.clips[i].audio);
    changed(); requestAnimationFrame(loop);
  }
  if (hashIn.has("t0") || hashIn.has("t1")) setView(+(hashIn.get("t0") ?? 0), +(hashIn.get("t1") ?? duration));
  else setView(0, duration);
  document.body.dataset.ready = "1";
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
