"""
Phase 2 step 2.1: fit every sentence of the evaluation set and score it with Whisper WER.

For each clip in ``assets/recordings/manifest.json``:

1. Level A analysis of the recording (``vocaltract.analysis.resynth``,
   default settings) gives engine frames and the engine's render.
2. The frames are moved onto the pipeline grid and fitted
   (:func:`vocaltract.phase2.utterance.fit`), with level A's frication as a
   fixed track.
3. Four conditions are scored against the reference transcript:
   - ``original``: the recording, the floor
   - ``level_a``: the reference engine's render of the level-A frames
   - ``init``: the pipeline's render of those frames before fitting
   - ``fit``: the pipeline's render of the fitted frames

   Whisper decodes with beam search (``--beam``, default 5). Greedy decoding
   swung between 24% and 86% WER on noise-seed variants of one render. The
   pipeline conditions are rendered with ``--seeds`` aspiration-noise
   realizations that differ from the one used in training, and their WER is
   the mean. That measures the parameters, not one noise sample. Level A and
   the original are deterministic, so they are scored once.
4. Harmonic-level error against the recording (below / above 4 kHz).

Outputs in ``--out`` (default ``out/phase2``), per clip: ``<id>.json`` (all
metrics), ``<id>_{original,level_a,init,fit}.wav``, ``<id>_params.npz``
(initial and fitted frames) and ``<id>_compare.html``. Across all clips:
``report.md`` with corpus WER (total errors / total words), the median
clip, the worst clip, and the acceptance check of step 2.1. Clips whose JSON
exists are not refitted (``--force`` refits them).

Usage::

    python -m vocaltract.phase2.harness                      # all ten, ~1.5 h on CPU
    python -m vocaltract.phase2.harness --clips 2961-960-0019 --steps 50
    python -m vocaltract.phase2.harness --report-only        # rebuild report.md
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch

from .utterance import NOISE_SEED, SAMPLE_RATE, FitConfig, Utterance, fit, prepare, to_rate

ROOT = Path(__file__).resolve().parents[3]
RECORDINGS = ROOT / "assets" / "recordings"
CONDITIONS = ("original", "level_a", "init", "fit")


def load_manifest(recordings: Path = RECORDINGS) -> List[dict]:
    return json.loads((recordings / "manifest.json").read_text())


def wer_counts(audio: np.ndarray, reference: str, model: str, beam: Optional[int]) -> dict:
    from ..analysis.intelligibility import score

    r = score(audio, SAMPLE_RATE, reference, model, beam)
    return {"wer": r.wer, "errors": r.substitutions + r.deletions + r.insertions,
            "substitutions": r.substitutions, "deletions": r.deletions, "insertions": r.insertions,
            "words": r.reference_words, "heard": r.hypothesis}


def mean_wer(runs: Sequence[dict]) -> dict:
    """Mean over noise realizations; ``runs`` kept for the transcripts."""
    return {"wer": float(np.mean([r["wer"] for r in runs])),
            "errors": float(np.mean([r["errors"] for r in runs])),
            "words": runs[0]["words"], "runs": list(runs)}


def peak(audio: np.ndarray, level: float = 0.9) -> np.ndarray:
    return audio * (level / max(np.abs(audio).max(), 1e-12))


def run_clip(entry: dict, cfg: FitConfig, args, out: Path) -> dict:
    from ..analysis.harmonic_error import HarmonicReference
    from ..analysis.resynth import resynthesize

    clip_id = entry["id"]
    audio, sr = sf.read(RECORDINGS / entry["file"], always_2d=True)
    audio = audio.mean(axis=1)
    reference = (RECORDINGS / f"{clip_id}.txt").read_text()

    level_a = resynthesize(audio, sr, clip_id)
    target = to_rate(audio, sr)
    init, frication, level_a_audio = prepare(level_a.frames, len(target), cfg.hop)
    n = len(init) * cfg.hop
    target = target[:n]
    print(f"{clip_id}: {len(target) / SAMPLE_RATE:.1f} s, {len(init)} frames, "
          f"{len(level_a.frames)} level-A frames", flush=True)

    result = fit(target, init, frication, cfg)
    for line in result.log:
        print(f"  {line}", flush=True)

    init_model = Utterance(None, cfg)  # untrained mapper, level A's frication at its level-A mix
    init_model.frication = result.model.frication
    seeds = [NOISE_SEED + 1 + s for s in range(args.seeds)]
    renders = {
        "original": [target],
        "level_a": [level_a_audio[:n]],
        "init": [init_model.render(result.initial_params, s) for s in seeds],
        "fit": [result.model.render(result.params, s) for s in seeds],
    }

    metrics: Dict[str, dict] = {}
    if args.asr != "none":
        for cond in CONDITIONS:
            runs = [wer_counts(x, reference, args.asr, args.beam) for x in renders[cond]]
            metrics[cond] = mean_wer(runs)
            print(f"  WER {cond:8s} {metrics[cond]['wer']:6.1%}", flush=True)

    harmonic = HarmonicReference(target, SAMPLE_RATE)
    for cond in ("level_a", "init", "fit"):
        below, above = harmonic.error(renders[cond][0])
        metrics.setdefault(cond, {})["harmonic_rms_db"] = [below, above]

    for cond in CONDITIONS:
        sf.write(out / f"{clip_id}_{cond}.wav", peak(renders[cond][0]), SAMPLE_RATE, subtype="PCM_16")
    np.savez(out / f"{clip_id}_params.npz", init=result.initial_params[0].numpy(),
             fit=result.params[0].numpy(), hop=cfg.hop, sample_rate=SAMPLE_RATE)
    if not args.no_compare:
        write_compare(out / f"{clip_id}_compare.html", clip_id, renders, metrics, result.params[0].numpy(),
                      cfg.hop)

    record = {
        "id": clip_id, "gender": entry.get("gender"), "median_f0_hz": entry.get("median_f0_hz"),
        "duration_s": len(target) / SAMPLE_RATE, "manifest_level_a_wer": entry.get("level_a_wer"),
        "config": asdict(cfg), "asr": args.asr, "beam": args.beam, "seeds": seeds,
        "loss": {"initial": result.history[0], "best": result.history[result.best_step],
                 "best_step": result.best_step},
        "bandwidths_hz": result.bandwidths_hz, "frication_gain": result.frication_gain,
        "fit_seconds": result.seconds, "metrics": metrics,
    }
    (out / f"{clip_id}.json").write_text(json.dumps(record, indent=1))
    return record


def write_compare(path: Path, clip_id: str, renders: dict, metrics: dict, params: np.ndarray,
                  hop: int) -> None:
    from ..analysis.compare import Clip, build_page

    def note(cond):
        return f"WER {metrics[cond]['wer']:.0%}" if "wer" in metrics.get(cond, {}) else ""

    clips = [Clip("original", renders["original"][0], SAMPLE_RATE, note("original")),
             Clip("level A (engine)", renders["level_a"][0], SAMPLE_RATE, note("level_a")),
             Clip("fit (pipeline)", renders["fit"][0], SAMPLE_RATE, note("fit"))]
    times = (np.arange(len(params)) + 0.5) * hop / SAMPLE_RATE
    frames = np.clip(np.round(params * 255), 0, 255).astype(np.uint8)
    path.write_text(build_page(clips, frames=frames, frame_times=times,
                               title=f"{clip_id}: original, level A, fit"), encoding="utf-8")


def corpus(records: Sequence[dict], cond: str) -> Optional[dict]:
    rows = [r["metrics"][cond] for r in records if "wer" in r["metrics"].get(cond, {})]
    if not rows:
        return None
    wers = [row["wer"] for row in rows]
    worst = int(np.argmax(wers))
    return {"wer": sum(row["errors"] for row in rows) / sum(row["words"] for row in rows),
            "median": float(np.median(wers)), "worst": wers[worst],
            "worst_id": [r["id"] for r in records if "wer" in r["metrics"].get(cond, {})][worst]}


def report(records: Sequence[dict]) -> str:
    """Markdown report: per-clip table, corpus summary and the step 2.1 acceptance check."""
    lines = ["# Phase 2.1 utterance fits", ""]
    if records:
        r0 = records[0]
        cfg = r0["config"]
        lines += [f"Settings: hop {cfg['hop']} samples ({1000 * cfg['hop'] / SAMPLE_RATE:.0f} ms), "
                  f"{cfg['steps']} steps, lr {cfg['lr']:g}; Whisper {r0['asr']}, beam {r0['beam']}, "
                  f"pipeline WER = mean over {len(r0['seeds'])} noise seeds.", ""]
    lines += ["| Clip | Sex | f0 | WER original | level A | init | fit | harmonic dB < / > 4 kHz: level A | fit |",
              "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda m: f"{m['wer']:.0%}" if m and "wer" in m else "–"
    hfmt = lambda m: "{:.1f} / {:.1f}".format(*m["harmonic_rms_db"]) if m and "harmonic_rms_db" in m else "–"
    for r in records:
        m = r["metrics"]
        lines.append(f"| {r['id']} | {r['gender']} | {r['median_f0_hz']:.0f} Hz | {fmt(m.get('original'))} | "
                     f"{fmt(m.get('level_a'))} | {fmt(m.get('init'))} | {fmt(m.get('fit'))} | "
                     f"{hfmt(m.get('level_a'))} | {hfmt(m.get('fit'))} |")
    lines += ["", "| Condition | Corpus WER | Median clip | Worst clip |", "|---|---|---|---|"]
    summary = {cond: corpus(records, cond) for cond in CONDITIONS}
    for cond, s in summary.items():
        if s:
            lines.append(f"| {cond} | {s['wer']:.1%} | {s['median']:.0%} | {s['worst']:.0%} ({s['worst_id']}) |")
    fit_s, la_s = summary["fit"], summary["level_a"]
    if fit_s and la_s:
        worse = [r["id"] for r in records
                 if r["metrics"]["fit"]["wer"] > r["metrics"]["level_a"]["wer"] + 1e-9]
        ok = fit_s["wer"] <= la_s["wer"] + 1e-9 and not worse
        lines += ["", f"**Acceptance (2.1):** fit corpus WER {fit_s['wer']:.1%} vs level A {la_s['wer']:.1%}; "
                  f"clips worse than their level A: {', '.join(worse) or 'none'}. "
                  f"{'PASS' if ok else 'FAIL'}."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*", help="clip ids (default: all in the manifest)")
    ap.add_argument("--out", type=Path, default=Path("out/phase2"))
    ap.add_argument("--hop", type=int, default=FitConfig.hop, help="160 = 10 ms, 320 = 20 ms")
    ap.add_argument("--steps", type=int, default=FitConfig.steps)
    ap.add_argument("--lr", type=float, default=FitConfig.lr)
    ap.add_argument("--no-frication", action="store_true", help="fit without level A's frication track")
    ap.add_argument("--asr", default="small.en", help="Whisper model, or 'none'")
    ap.add_argument("--beam", type=int, default=5, help="beam size (0: greedy)")
    ap.add_argument("--seeds", type=int, default=3, help="noise realizations scored per pipeline render")
    ap.add_argument("--no-compare", action="store_true", help="skip the compare pages")
    ap.add_argument("--force", action="store_true", help="refit clips that already have results")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    args.beam = args.beam or None

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    entries = [e for e in manifest if not args.clips or e["id"] in args.clips]
    if args.clips and len(entries) != len(args.clips):
        known = {e["id"] for e in manifest}
        print(f"error: unknown clip(s): {', '.join(c for c in args.clips if c not in known)}", file=sys.stderr)
        return 2
    cfg = FitConfig(hop=args.hop, steps=args.steps, lr=args.lr, frication=not args.no_frication)

    records = []
    for entry in entries:
        path = args.out / f"{entry['id']}.json"
        if path.exists() and not args.force or args.report_only:
            if path.exists():
                records.append(json.loads(path.read_text()))
            continue
        records.append(run_clip(entry, cfg, args, args.out))
    text = report(records)
    (args.out / "report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
