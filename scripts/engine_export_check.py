"""
Play Phase 2 fits on the reference engine: the WER of the 13 bytes, not of the pipeline.

For each clip in a harness output folder (``out/phase2``), the initial and fitted parameter frames
(``<id>_params.npz``, normalized, on the pipeline grid) are moved to the engine's frame instants
((k + 1) * 9.8 ms, linear interpolation), the fit's ``fa`` is multiplied by the learned frication gain,
everything is rounded to bytes, rendered by ``vocaltract.reference`` and scored with Whisper (beam 5).
The engine is deterministic, so each render is scored once.

It reads the harness's results, so run ``python -m vocaltract.phase2.harness`` first (about 1.5 h), or point it
at a checkout that has them.

Usage::

    python scripts/engine_export_check.py out/phase2 assets/recordings out/phase2_engine
    python scripts/engine_export_check.py ../TTS_VocalTract/out/phase2 assets/recordings out/phase2_engine
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from vocaltract.analysis.intelligibility import score
from vocaltract.analysis.resynth import HOP_S, render_frames
from vocaltract.reference.engine import FA


def to_engine_frames(params: np.ndarray, hop: int, sr: int, fric_gain: float = 1.0) -> np.ndarray:
    """(N, 13) normalized pipeline frames -> (K, 13) uint8 engine frames."""
    centres = (np.arange(len(params)) + 0.5) * hop / sr
    knots = (np.arange(int(centres[-1] / HOP_S)) + 1) * HOP_S
    f = np.stack([np.interp(knots, centres, params[:, i]) for i in range(params.shape[1])], 1) * 255
    f[:, FA] *= fric_gain
    return np.clip(np.round(f), 0, 255).astype(np.uint8)


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    out, rec, dst = (Path(a) for a in sys.argv[1:4])
    results = sorted(p for p in out.glob("*.json") if (out / f"{p.stem}_params.npz").exists())
    if not results:
        print(f"error: no harness results (<id>.json + <id>_params.npz) in {out}; "
              "run `python -m vocaltract.phase2.harness` first", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for jp in results:
        r = json.loads(jp.read_text())
        cid = r["id"]
        z = np.load(out / f"{cid}_params.npz")
        hop, sr = int(z["hop"]), int(z["sample_rate"])
        ref = (rec / f"{cid}.txt").read_text()
        row = {"id": cid, "level_a": r["metrics"]["level_a"]["wer"], "pipeline_fit": r["metrics"]["fit"]["wer"]}
        for cond in ("init", "fit"):
            frames = to_engine_frames(z[cond], hop, sr, r["frication_gain"] if cond == "fit" else 1.0)
            audio = signal.resample_poly(render_frames(frames).astype(np.float64) / 2**31, 4, 5)
            sf.write(dst / f"{cid}_{cond}_engine.wav", audio / max(np.abs(audio).max(), 1e-9) * 0.9, 16000,
                     subtype="PCM_16")
            w = score(audio, 16000, ref, "small.en", 5)
            row[f"engine_{cond}"] = w.wer
            row[f"engine_{cond}_errors"] = w.substitutions + w.deletions + w.insertions
            row["words"] = w.reference_words
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    (dst / "engine_wer.json").write_text(json.dumps(rows, indent=1))
    words = sum(r["words"] for r in rows)
    for cond in ("init", "fit"):
        print(f"engine {cond}: corpus WER {sum(r[f'engine_{cond}_errors'] for r in rows) / words:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
