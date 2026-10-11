"""
Side-by-side spectrogram comparison of audio files, as a self-contained HTML page.

Writes one HTML file (no server, no network) with:

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
- a **difference** pane (2nd file minus 1st, in dB) when comparing two files

Usage::

    python -m vocaltract.analysis.compare original.wav resynth.wav \\
        --frames out/resynth/x_frames.csv --words -o out/compare.html

``vocaltract.analysis.resynth`` writes one of these per input automatically
(``<name>_compare.html``). ``--sv-dir`` also exports the overlays as CSV
layers for Sonic Visualiser (File > Import Annotation Layer).
"""

import argparse
import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..params import FORMANT_SCALE, PARAM_NAMES

# name -> (window seconds, hop seconds, frequency rows)
RESOLUTIONS = {"wide": (0.005, 0.002, 256), "narrow": (0.035, 0.005, 512)}
FLOOR_DB = -100.0          # stored range; the page's slider picks the display floor
DIFF_RANGE_DB = 40.0       # difference pane spans +/- this
MAX_COLUMNS = 12_000       # longer files get a coarser hop
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
    sit at the floor.
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


def encode_diff(db_a: np.ndarray, db_b: np.ndarray, quiet_db: float = -70.0) -> str:
    """8-bit PNG of ``b - a`` in dB, 128 = equal; silent in both -> equal."""
    d = np.where(np.maximum(db_a, db_b) < quiet_db, 0.0, db_b - db_a)
    gray = 128 + np.clip(d / DIFF_RANGE_DB, -1, 1) * 127
    return _png_b64(np.round(gray).astype(np.uint8))


def _wav_b64(audio: np.ndarray, sr: int) -> str:
    import soundfile as sf

    x = np.asarray(audio, dtype=np.float64)
    peak = np.abs(x).max()
    buf = io.BytesIO()
    sf.write(buf, x * (0.9 / peak) if peak > 0 else x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _lut(name: str) -> list[list[int]]:
    from matplotlib import colormaps

    return (np.asarray(colormaps[name](np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(int).tolist()


def _clean(values, digits: int = 1) -> list:
    """JSON-friendly list: NaN -> null, rounded."""
    return [None if v is None or not np.isfinite(v) else round(float(v), digits) for v in values]


# ---------------------------------------------------------------------------
# page
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
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.stack([data[n] for n in PARAM_NAMES], axis=1), data["time_s"]


def build_page(clips: list[Clip], *, frames: np.ndarray | None = None,
               frame_times: np.ndarray | None = None, words: list[list[dict]] | None = None,
               praat: bool = True, fmax: float | None = None, title: str = "Spectrogram comparison") -> str:
    """Render the comparison page as an HTML string."""
    if not clips:
        raise ValueError("need at least one clip")
    fmax = float(fmax or min(c.sr for c in clips) / 2)
    duration = max(c.duration for c in clips)

    grid, specs = {}, [dict() for _ in clips]
    raw = [dict() for _ in clips]
    for res, (window, hop, n_rows) in RESOLUTIONS.items():
        hop = max(hop, duration / MAX_COLUMNS)
        n_cols = int(np.ceil(duration / hop)) + 1
        grid[res] = {"hop": hop, "cols": n_cols, "rows": n_rows, "window": window}
        for i, c in enumerate(clips):
            raw[i][res] = spectrogram(c.audio, c.sr, window, hop, fmax, n_cols, n_rows)
            specs[i][res] = encode_db(raw[i][res])

    data = {
        "title": title,
        "duration": duration,
        "fmax": fmax,
        "floorStored": FLOOR_DB,
        "diffRange": DIFF_RANGE_DB,
        "grid": grid,
        "lut": {"magma": _lut("magma"), "diverging": _lut("RdBu_r")},
        "clips": [{
            "name": c.name, "note": c.note, "sr": c.sr, "duration": c.duration,
            "audio": _wav_b64(c.audio, c.sr), "spec": specs[i],
            "words": (words[i] if words and i < len(words) and words[i] else []),
            "praat": praat_tracks(c) if praat else None,
        } for i, c in enumerate(clips)],
        "diff": ({res: encode_diff(raw[0][res], raw[1][res]) for res in RESOLUTIONS}
                 if len(clips) == 2 else None),
        "frames": frames_overlay(frames, frame_times) if frames is not None else None,
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (_TEMPLATE.replace("__TITLE__", _html_escape(title))
            .replace("__DATA__", blob))


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Side-by-side spectrogram comparison page")
    ap.add_argument("inputs", nargs="+", type=Path, help="audio files, top to bottom")
    ap.add_argument("-o", "--out", type=Path, default=Path("out") / "compare.html")
    ap.add_argument("--names", help="comma-separated display names")
    ap.add_argument("--frames", type=Path, help="resynth <name>_frames.csv to overlay")
    ap.add_argument("--words", action="store_true", help="Whisper word timings per file")
    ap.add_argument("--asr", default="small.en", help="Whisper model for --words")
    ap.add_argument("--no-praat", action="store_true", help="skip Praat pitch/formant tracks")
    ap.add_argument("--start", type=float, default=0.0, help="trim: start time (s), all files")
    ap.add_argument("--duration", type=float, help="trim: length (s)")
    ap.add_argument("--fmax", type=float, help="top frequency (default: lowest Nyquist)")
    ap.add_argument("--title", help="page title")
    ap.add_argument("--sv-dir", type=Path, help="also export overlays as Sonic Visualiser CSV layers")
    args = ap.parse_args(argv)

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
:root{--bg:#f5f5f3;--panel:#ffffff;--text:#1c1c1e;--muted:#6a6a70;--border:#d8d8dd;--accent:#2f6fde;--axis:#5a5a60;--grid:rgba(0,0,0,.08)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121214;--panel:#1b1b1e;--text:#ececef;--muted:#9898a0;--border:#323237;--accent:#6aa0ff;--axis:#a2a2aa;--grid:rgba(255,255,255,.08)}}
:root[data-theme="dark"]{--bg:#121214;--panel:#1b1b1e;--text:#ececef;--muted:#9898a0;--border:#323237;--accent:#6aa0ff;--axis:#a2a2aa;--grid:rgba(255,255,255,.08)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--border);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center}
header h1{font-size:15px;margin:0 8px 0 0;font-weight:650}
.group{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
label{color:var(--muted);font-size:13px;display:flex;gap:5px;align-items:center}
select,input[type=range]{accent-color:var(--accent)}
select{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:3px 6px;font:inherit;font-size:13px}
button{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 10px;font:inherit;font-size:13px;cursor:pointer}
button.active{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
button .k{color:var(--muted);font-size:11px;margin-left:4px}
main{padding:4px 0 8px}
.pane{margin:8px 16px;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.pane .head{display:flex;gap:10px;align-items:baseline;padding:6px 10px 2px;font-size:13px}
.pane .head b{font-weight:650}
.pane .head .note{color:var(--muted)}
.pane .head .legend{margin-left:auto;color:var(--muted);font-size:12px;display:flex;gap:10px;flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px;box-shadow:0 0 0 1px var(--border)}
canvas{display:block;width:100%;touch-action:none;cursor:grab}
canvas.dragging{cursor:grabbing}
.help{color:var(--muted);font-size:12px;margin:4px 16px 18px}
.hidden{display:none}
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
</header>
<main id="panes"></main>
<p class="help">URL options: <code>#t0=2&amp;t1=4&amp;res=narrow&amp;floor=-60&amp;fmax=5000&amp;diff=1&amp;praat=1&amp;words=0</code> &middot; Wheel or pinch: zoom &middot; drag: pan &middot; double-click: whole file &middot; click: move playhead &middot; <b>space</b>: play/pause &middot; <b>1&ndash;9</b>: switch file at the same moment (A/B) &middot; <b>&larr;/&rarr;</b>: step 0.5 s</p>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const L = 46, R = 10;                      // plot margins (css px)
const SPEC_H = 220, WORDS_H = 22, AXIS_H = 18, LANE_H = 110;
const CLIP_COLORS = ["#ffb000", "#35d0ff", "#ff5fa2", "#8ee05a", "#c08bff", "#ff7b39"];
const state = {t0: 0, t1: D.duration, res: "wide", floor: -70, fmax: D.fmax, active: 0, cursor: 0,
               ovFrames: true, ovPraat: false, ovWords: true, ovDiff: false};
const $ = id => document.getElementById(id);
document.title = D.title; $("title").textContent = D.title;

// ---------- audio ----------
const audios = D.clips.map(c => { const a = new Audio("data:audio/wav;base64," + c.audio); a.preload = "auto"; return a; });
const playing = () => !audios[state.active].paused;
function nowT() { return playing() ? audios[state.active].currentTime : state.cursor; }
function switchTo(i, play) {
  const t = nowT(), wasPlaying = playing();
  audios[state.active].pause();
  state.active = i; state.cursor = Math.min(t, D.clips[i].duration);
  audios[i].currentTime = state.cursor;
  if (play ?? wasPlaying) audios[i].play();
  updateButtons(); dirty = true;
}
function togglePlay() {
  const a = audios[state.active];
  if (a.paused) { if (state.cursor >= D.clips[state.active].duration - 0.01) state.cursor = 0; a.currentTime = state.cursor; a.play(); }
  else { a.pause(); state.cursor = a.currentTime; }
  updateButtons();
}
function seek(t) {
  state.cursor = Math.max(0, Math.min(t, D.clips[state.active].duration));
  audios[state.active].currentTime = state.cursor; dirty = true;
}
audios.forEach((a, i) => { a.addEventListener("ended", () => { state.cursor = D.clips[i].duration; updateButtons(); dirty = true; }); });

const btns = D.clips.map((c, i) => {
  const b = document.createElement("button");
  b.innerHTML = `<span class="sw" style="background:${CLIP_COLORS[i % 6]}"></span><span class="lbl"></span><span class="k">${i + 1}</span>`;
  b.onclick = () => { if (state.active === i) togglePlay(); else switchTo(i, true); };
  $("players").appendChild(b); return b;
});
function updateButtons() {
  btns.forEach((b, i) => {
    const on = i === state.active;
    b.classList.toggle("active", on);
    b.querySelector(".lbl").textContent = (on && playing() ? "❚❚ " : "▶ ") + D.clips[i].name;
  });
}

// ---------- spectrogram images ----------
function decodeGray(b64) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
      const cx = cv.getContext("2d"); cx.drawImage(img, 0, 0);
      const px = cx.getImageData(0, 0, img.width, img.height).data;
      const g = new Uint8Array(img.width * img.height);
      for (let i = 0; i < g.length; i++) g[i] = px[i * 4];
      resolve({w: img.width, h: img.height, g});
    };
    img.src = "data:image/png;base64," + b64;
  });
}
function colorize(G, lut, map) {
  const cv = document.createElement("canvas"); cv.width = G.w; cv.height = G.h;
  const cx = cv.getContext("2d"), im = cx.createImageData(G.w, G.h), d = im.data;
  const table = new Array(256); for (let v = 0; v < 256; v++) table[v] = lut[map(v)];
  for (let i = 0, j = 0; i < G.g.length; i++, j += 4) { const c = table[G.g[i]]; d[j] = c[0]; d[j + 1] = c[1]; d[j + 2] = c[2]; d[j + 3] = 255; }
  cx.putImageData(im, 0, 0); return cv;
}
const gray = D.clips.map(() => ({})), diffGray = {};
const colored = D.clips.map(() => ({})), diffColored = {};
function specMap(v) {
  const db = D.floorStored + v / 255 * -D.floorStored;
  return Math.max(0, Math.min(255, Math.round((db - state.floor) / -state.floor * 255)));
}
function recolor() {
  D.clips.forEach((c, i) => { for (const r in gray[i]) colored[i][r] = colorize(gray[i][r], D.lut.magma, specMap); });
  for (const r in diffGray) diffColored[r] = colorize(diffGray[r], D.lut.diverging, v => v);
  dirty = true;
}

// ---------- panes ----------
const panes = [];
function makePane(kind, idx, title, legend) {
  const div = document.createElement("div"); div.className = "pane";
  const head = document.createElement("div"); head.className = "head";
  head.innerHTML = `<b></b><span class="note"></span><span class="legend"></span>`;
  head.querySelector("b").textContent = title;
  if (kind === "spec" && D.clips[idx].note) head.querySelector(".note").textContent = D.clips[idx].note;
  if (legend) head.querySelector(".legend").innerHTML = legend;
  const cv = document.createElement("canvas");
  div.appendChild(head); div.appendChild(cv); $("panes").appendChild(div);
  const p = {kind, idx, div, cv, ctx: cv.getContext("2d")};
  attachInput(p); panes.push(p); return p;
}
const sw = (c, t) => `<span><span class="sw" style="background:${c}"></span>${t}</span>`;
const specLegend = (D.frames ? sw("#35e0ff", "frames formants") : "") + (D.clips.some(c => c.praat) ? sw("#ffffff", "Praat formants") : "");
D.clips.forEach((c, i) => makePane("spec", i, `${i + 1}. ${c.name}`, specLegend));
const diffPane = D.diff ? makePane("diff", 0, `difference: ${D.clips[1].name} − ${D.clips[0].name}`,
  sw("#c0392b", `${D.clips[1].name} louder`) + sw("#2e6db4", `${D.clips[0].name} louder`) + `<span>±${D.diffRange} dB</span>`) : null;
const hasPraat = D.clips.some(c => c.praat);
const pitchPane = (hasPraat || D.frames) ? makePane("pitch", 0, "pitch",
  D.clips.map((c, i) => c.praat ? sw(CLIP_COLORS[i % 6], `Praat f0: ${c.name}`) : "").join("") + (D.frames ? sw("var(--text)", "frames gp") : "")) : null;
const ampPane = D.frames ? makePane("amp", 0, "source amplitudes (frames)",
  sw("#ff9f1c", "ga voice") + sw("#2ec4b6", "aa aspiration") + sw("#b388ff", "fa frication")) : null;
if (!D.frames) { $("lblFrames").classList.add("hidden"); }
if (!hasPraat) { $("lblPraat").classList.add("hidden"); }
if (!D.clips.some(c => c.words.length)) { $("lblWords").classList.add("hidden"); state.ovWords = false; }
if (!D.diff) { $("lblDiff").classList.add("hidden"); }
if (diffPane) diffPane.div.classList.add("hidden");

function paneHeight(p) {
  if (p.kind === "spec") return SPEC_H + (state.ovWords && D.clips[p.idx].words.length ? WORDS_H : 0) + AXIS_H;
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
function drawImageView(p, img, top, H) {
  const g = D.grid[state.res], c = p.ctx;
  const rows = img.height, shown = Math.max(2, Math.round(rows * state.fmax / D.fmax));
  const sx = state.t0 / g.hop, sw = (state.t1 - state.t0) / g.hop;
  c.imageSmoothingEnabled = true;
  c.drawImage(img, sx, rows - shown, sw, shown, L, top, p.w - L - R, H);
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
  const c = p.ctx; c.strokeStyle = color; c.lineWidth = width || 1.4; c.beginPath(); let pen = false;
  for (let k = 0; k < ts.length; k++) {
    const v = vs[k];
    if (v === null) { pen = false; continue; }
    const x = tx(p, ts[k]), y = scaleY(v);
    if (pen) c.lineTo(x, y); else c.moveTo(x, y); pen = true;
  }
  c.stroke(); c.lineWidth = 1;
}
function drawPlayhead(p, top, H) {
  const t = nowT(); if (t < state.t0 || t > state.t1) return;
  const c = p.ctx, x = tx(p, t);
  c.strokeStyle = "rgba(255,255,255,.9)"; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, top); c.lineTo(x, top + H); c.stroke();
  c.strokeStyle = "rgba(0,0,0,.5)"; c.lineWidth = 1; c.beginPath(); c.moveTo(x + 1.5, top); c.lineTo(x + 1.5, top + H); c.stroke();
}
function drawSpec(p) {
  const c = p.ctx, clip = D.clips[p.idx], top = 0, H = SPEC_H;
  c.clearRect(0, 0, p.w, p.h);
  const img = colored[p.idx][state.res]; if (!img) return;
  drawImageView(p, img, top, H); drawFreqAxis(p, top, H);
  const fy = f => top + H * (1 - f / state.fmax);
  if (state.ovPraat && clip.praat) for (const F of clip.praat.F) dots(p, clip.praat.t, F, null, top, H, "rgba(255,255,255,.75)", 1.2, fy);
  if (state.ovFrames && D.frames) for (const F of D.frames.F) dots(p, D.frames.t, F, D.frames.on, top, H, "rgba(53,224,255,.85)", 1.4, fy);
  if (p.idx === state.active) { c.strokeStyle = CLIP_COLORS[p.idx % 6]; c.lineWidth = 3; c.beginPath(); c.moveTo(L - 2, top); c.lineTo(L - 2, top + H); c.stroke(); c.lineWidth = 1; }
  let y = top + H;
  if (state.ovWords && clip.words.length) { drawWords(p, clip.words, y, WORDS_H); y += WORDS_H; }
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
  const img = diffColored[state.res]; if (!img) return;
  drawImageView(p, img, 0, SPEC_H); drawFreqAxis(p, 0, SPEC_H);
  drawPlayhead(p, 0, SPEC_H); drawTimeAxis(p, SPEC_H);
}
let pitchRange = null;
function drawPitch(p) {
  const c = p.ctx, top = 4, H = LANE_H - 8; c.clearRect(0, 0, p.w, p.h);
  if (!pitchRange) {
    const all = [];
    D.clips.forEach(cl => cl.praat && cl.praat.f0.forEach(v => v && all.push(v)));
    if (D.frames) D.frames.gp.forEach(v => v && all.push(v));
    all.sort((a, b) => a - b);
    const lo = all.length ? all[Math.floor(all.length * .02)] : 80, hi = all.length ? all[Math.floor(all.length * .98)] : 300;
    pitchRange = [Math.max(40, lo / 1.25), Math.min(1000, hi * 1.25)];
  }
  const [lo, hi] = pitchRange, py = f => top + H * (1 - Math.log(f / lo) / Math.log(hi / lo));
  c.fillStyle = css("--grid"); c.fillRect(L, top, p.w - L - R, H);
  c.fillStyle = css("--axis"); c.font = "11px system-ui"; c.textAlign = "right"; c.textBaseline = "middle";
  for (const f of [50, 75, 100, 150, 200, 300, 400, 600, 800]) if (f >= lo && f <= hi) { c.fillText(f, L - 6, py(f)); }
  c.textBaseline = "alphabetic";
  D.clips.forEach((cl, i) => { if (cl.praat) dots(p, cl.praat.t, cl.praat.f0, null, top, H, CLIP_COLORS[i % 6], 1.6, py); });
  if (D.frames) line(p, D.frames.t, D.frames.gp, top, H, css("--text"), py, 1.2);
  drawPlayhead(p, top, H); drawTimeAxis(p, LANE_H);
}
function drawAmp(p) {
  const c = p.ctx, top = 4, H = LANE_H - 8; c.clearRect(0, 0, p.w, p.h);
  c.fillStyle = css("--grid"); c.fillRect(L, top, p.w - L - R, H);
  const ay = v => top + H * (1 - v / 255);
  c.fillStyle = css("--axis"); c.font = "11px system-ui"; c.textAlign = "right"; c.textBaseline = "middle";
  for (const v of [0, 128, 255]) c.fillText(v, L - 6, ay(v));
  c.textBaseline = "alphabetic";
  line(p, D.frames.t, D.frames.ga, top, H, "#ff9f1c", ay);
  line(p, D.frames.t, D.frames.aa, top, H, "#2ec4b6", ay);
  line(p, D.frames.t, D.frames.fa, top, H, "#b388ff", ay);
  drawPlayhead(p, top, H); drawTimeAxis(p, LANE_H);
}
function drawAll() {
  for (const p of panes) {
    if (p.div.classList.contains("hidden")) continue;
    if (p.kind === "spec") drawSpec(p); else if (p.kind === "diff") drawDiff(p);
    else if (p.kind === "pitch") drawPitch(p); else drawAmp(p);
  }
}
let dirty = true;
function loop() {
  if (playing()) {
    const t = audios[state.active].currentTime, span = state.t1 - state.t0;
    if (t > state.t1 || t < state.t0) { state.t0 = Math.max(0, Math.min(t - span * 0.1, D.duration - span)); state.t1 = state.t0 + span; }
    dirty = true;
  }
  if (dirty) { dirty = false; drawAll(); }
  requestAnimationFrame(loop);
}

// ---------- interaction ----------
function setView(t0, t1) {
  const span = Math.max(0.05, Math.min(D.duration, t1 - t0));
  t0 = Math.max(0, Math.min(t0, D.duration - span));
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
  p.cv.addEventListener("dblclick", () => setView(0, D.duration));
}
window.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (/^Digit[1-9]$/.test(e.code)) { const i = +e.code.slice(5) - 1; if (i < D.clips.length) switchTo(i); }
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
$("ovDiff").onchange = e => { state.ovDiff = e.target.checked; diffPane.div.classList.toggle("hidden", !state.ovDiff); resize(); };
window.addEventListener("resize", resize);

// ---------- URL options (#t0=..&t1=..&res=narrow&floor=-60&fmax=5000&diff=1&praat=1&frames=0&words=0) ----------
function applyHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has("res") && D.grid[h.get("res")]) { state.res = h.get("res"); $("res").value = state.res; }
  if (h.has("floor")) { state.floor = Math.max(-100, Math.min(-10, +h.get("floor"))); $("floor").value = state.floor; $("floorv").textContent = state.floor + " dB"; }
  if (h.has("fmax")) { state.fmax = Math.min(D.fmax, +h.get("fmax")); if (![...$("fmax").options].some(o => +o.value === state.fmax)) $("fmax").add(new Option(state.fmax + " Hz", state.fmax)); $("fmax").value = state.fmax; }
  for (const [key, id] of [["frames", "ovFrames"], ["praat", "ovPraat"], ["words", "ovWords"], ["diff", "ovDiff"]])
    if (h.has(key)) { state[id] = h.get(key) === "1"; $(id).checked = state[id]; }
  if (diffPane) diffPane.div.classList.toggle("hidden", !state.ovDiff);
  if (h.has("t0") || h.has("t1")) setView(+(h.get("t0") ?? 0), +(h.get("t1") ?? D.duration));
}

// ---------- start ----------
(async () => {
  applyHash(); updateButtons(); resize();
  for (let i = 0; i < D.clips.length; i++) for (const r in D.clips[i].spec) gray[i][r] = await decodeGray(D.clips[i].spec[r]);
  if (D.diff) for (const r in D.diff) diffGray[r] = await decodeGray(D.diff[r]);
  recolor(); requestAnimationFrame(loop);
  document.body.dataset.ready = "1";
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
