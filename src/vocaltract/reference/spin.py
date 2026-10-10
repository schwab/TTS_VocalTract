"""
Shared pieces for porting Spin programs that drive VocalTract objects.

A Spin demo like ``SingingDemoSeven.spin`` has three parts, and only the
middle one is demo-specific:

1. **data** in the .spin file -- ``CON`` constants/enums (notes, flags) and
   ``DAT`` tables (the score, volumes, pan angles). :func:`parse_spin` reads
   them, so ports never retype the score.
2. **code** -- the PUB/PRI methods. These are ported by hand onto
   :class:`VocalTractProgram`, which mirrors the VocalTract object API
   (``set``/``get`` on the 13-byte parameter block, ``go``, ``set_pace``,
   ``set_attenuation``) and records every ``go()`` as a :class:`QueuedFrame`.
   :class:`SpinRandom` reproduces Spin's ``?var`` operator for demos that
   randomize timing.
3. **output** -- :func:`render_voices`, :func:`mix_stereo` and
   :func:`run_demo_cli` turn the frames into WAVs the same way for every demo.
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .engine import PARAM_COUNT, SAMPLE_RATE, QueuedFrame, render, stepsize

SPIN_DIR = Path(__file__).resolve().parents[3] / "spin"


# ---------------------------------------------------------------------------
# Spin operators
# ---------------------------------------------------------------------------

def random_forward(value: int) -> int:
    """Spin ``?var``: 32 iterations of the interpreter's LFSR (0 seeds as 1).

    Port of Gear's InterpretedCog 'Forward random' (from Cluso99's Spin
    bytecode document). Returns the new unsigned 32-bit value.
    """
    result = (value & 0xFFFFFFFF) or 1
    for _ in range(32):
        parity = result ^ (result >> 1) ^ (result >> 2) ^ (result >> 4)
        result >>= 1
        if parity & 1:
            result |= 0x80000000
    return result


def signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


class SpinRandom:
    """A Spin long used with ``?var``; ``next()`` returns the signed new value.

    Spin's ``~>`` (arithmetic shift right) is plain ``>>`` on the result.
    """

    def __init__(self, seed: int = 0):
        self.value = seed & 0xFFFFFFFF

    def next(self) -> int:
        self.value = random_forward(self.value)
        return signed32(self.value)


# ---------------------------------------------------------------------------
# .spin data sections
# ---------------------------------------------------------------------------

@dataclass
class SpinSource:
    """CON constants and DAT tables of a .spin file."""

    constants: dict[str, int] = field(default_factory=dict)
    tables: dict[str, list] = field(default_factory=dict)  # ints, or str for '@label'

    def notes(self, label: str) -> list[int]:
        """A zero-terminated byte table (a score part), without the terminator."""
        seq = self.tables[label]
        return seq[:seq.index(0)]


def eval_expr(expr: str, names: dict[str, int]) -> int:
    """Evaluate a Spin constant expression of +/- terms: names, 12, $5C00, %1010."""
    total, sign = 0, 1
    for token in re.findall(r"[+-]|[^\s+-]+", expr):
        if token in "+-":
            sign = 1 if token == "+" else -1
            continue
        token = token.replace("_", "") if token[0] in "$%0123456789" else token
        if token.startswith("$"):
            value = int(token[1:], 16)
        elif token.startswith("%"):
            value = int(token[1:], 2)
        elif token.isdigit():
            value = int(token)
        else:
            value = names[token]
        total += sign * value
        sign = 1
    return total


_ENUM_ITEM = re.compile(r"^(\w+)(?:\[(\d+)\])?$")


def _parse_con(con: str) -> dict[str, int]:
    """'name = expr' constants and '#start, a, b[n], ...' enumerations."""
    names: dict[str, int] = {}
    counter = None                      # active enumeration value, if any
    for line in con.splitlines():
        line = line.split("'", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^(\w+)\s*=\s*(.+)$", line)
        if m:
            counter = None
            try:
                names[m[1]] = eval_expr(m[2], names)
            except KeyError:            # e.g. _clkmode = xtal1 + pll16x
                pass
            continue
        items = [i.strip() for i in line.split(",") if i.strip()]
        if items and items[0].startswith("#"):
            counter = eval_expr(items.pop(0)[1:], names)
        if counter is None:
            continue
        for item in items:
            m = _ENUM_ITEM.match(item)
            if not m:
                break
            names[m[1]] = counter
            counter += int(m[2]) if m[2] is not None else 1
    return names


def read_spin_text(path: Path) -> str:
    """Read a .spin file: the Propeller Tool saves UTF-16 (with BOM), others UTF-8/Latin-1."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_spin(path: Path) -> SpinSource:
    text = read_spin_text(path)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.S)       # doc comments
    text = re.sub(r"\{[^{}]*\}", "", text)                    # inline comments
    sections = re.split(r"^(CON|VAR|OBJ|PUB|PRI|DAT)\b", text, flags=re.M)
    con = "".join(body for kind, body in zip(sections[1::2], sections[2::2]) if kind == "CON")
    dat = "".join(body for kind, body in zip(sections[1::2], sections[2::2]) if kind == "DAT")

    source = SpinSource(constants=_parse_con(con))
    label = None
    for line in dat.splitlines():
        line = line.split("'", 1)[0].rstrip()
        m = re.match(r"^(\w+)?\s+(byte|word|long)\s+(.+)$", line)
        if not m:
            continue
        if m[1]:
            label = m[1]
            source.tables[label] = []
        for item in m[3].split(","):
            item = item.strip()
            source.tables[label].append(
                item[1:] if item.startswith("@") else eval_expr(item, source.constants))
    return source


# ---------------------------------------------------------------------------
# The VocalTract object API, recording frames
# ---------------------------------------------------------------------------

class VocalTractProgram:
    """Base for Spin program ports: N tracts' parameter bytes plus their go() frames."""

    def __init__(self, n_tracts: int = 4):
        self.n_tracts = n_tracts
        self.vt = [[0] * PARAM_COUNT for _ in range(n_tracts)]  # 'byte vt[n*13]'
        self.pace = [100] * n_tracts
        self.attenuation = [0] * n_tracts
        self.frames: list[QueuedFrame] = []

    def set(self, tract: int, parameter: int, value: int) -> None:
        self.vt[tract][parameter] = value & 0xFF            # byte store truncates

    def get(self, tract: int, parameter: int) -> int:
        return self.vt[tract][parameter]

    def go(self, tract: int, time: int) -> None:
        self.frames.append(QueuedFrame(tract, tuple(self.vt[tract]),
                                       stepsize(time, self.pace[tract])))

    def set_pace(self, tract: int, percentage: int) -> None:
        self.pace[tract] = percentage

    def set_attenuation(self, tract: int, level: int) -> None:
        self.attenuation[tract] = level


# ---------------------------------------------------------------------------
# Rendering and output
# ---------------------------------------------------------------------------

def render_voices(frames: Sequence[QueuedFrame], n_tracts: int, sync: bool = False,
                  attenuation: Sequence[int] | int = 0) -> np.ndarray:
    """Render to ``(n_tracts, samples)`` raw int32 voices.

    ``sync=False`` co-simulates one Spin cog feeding every tract in program
    order (the parent blocks on whichever queue is full -- faithful timing).
    ``sync=True`` gives each tract its own unblocked feeder.
    """
    if not sync:
        return render(frames, n_tracts=n_tracts, attenuation=attenuation)
    att = [attenuation] * n_tracts if isinstance(attenuation, int) else list(attenuation)
    voices = [render([QueuedFrame(0, f.params, f.step) for f in frames if f.tract == t],
                     n_tracts=1, attenuation=att[t])[0] for t in range(n_tracts)]
    out = np.zeros((n_tracts, max(len(v) for v in voices)), dtype=np.int32)
    for t, v in enumerate(voices):
        out[t, :len(v)] = v
    return out


def mix_stereo(voices: np.ndarray, angles: Sequence[int]) -> np.ndarray:
    """Equal-power pan raw int32 voices to float stereo ``(samples, 2)``.

    ``angles`` are StereoSpatializer angles: $0000 = left .. $FFFF = right.
    (The spatializer's echoes/depth aren't modelled -- its source isn't here.)
    """
    pos = np.asarray(angles, dtype=np.float64)[:, None] / 0xFFFF * (np.pi / 2)
    v = voices.astype(np.float64) / 2**31
    return np.stack([(v * np.cos(pos)).sum(0), (v * np.sin(pos)).sum(0)], axis=1)


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.abs(audio).max()
    return audio * (peak / m) if m > 0 else audio


@dataclass
class DemoRender:
    """What a demo port hands to :func:`run_demo_cli`."""

    frames: list[QueuedFrame]
    voice_names: list[str]
    angles: list[int]
    attenuation: list[int]


def run_demo_cli(description: str, default_spin: Path, default_out: Path,
                 build: Callable[[Path], DemoRender], argv: list[str] | None = None) -> None:
    """Common command line: render a demo to a stereo WAV (+ optional stems)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--spin", type=Path, default=default_spin, help="path to the .spin file")
    ap.add_argument("-o", "--out", type=Path, default=default_out, help="output WAV path")
    ap.add_argument("--stems", action="store_true", help="also write one mono WAV per voice")
    ap.add_argument("--sync", action="store_true",
                    help="feed each voice independently instead of the demo's single "
                         "round-robin cog")
    ap.add_argument("--voices", help="comma-separated tracts to mix, e.g. '3' (default: all)")
    args = ap.parse_args(argv)

    import soundfile as sf

    demo = build(args.spin)
    n = len(demo.voice_names)
    voices = render_voices(demo.frames, n, sync=args.sync, attenuation=demo.attenuation)
    keep = [int(v) for v in args.voices.split(",")] if args.voices else list(range(n))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stereo = mix_stereo(voices[keep], [demo.angles[i] for i in keep])
    sf.write(args.out, normalize(stereo), SAMPLE_RATE, subtype="PCM_16")
    print(f"{len(demo.frames)} frames, {voices.shape[1] / SAMPLE_RATE:.1f} s -> {args.out}")

    if args.stems:
        for i in keep:
            stem = args.out.with_name(f"{args.out.stem}_{i}_{demo.voice_names[i]}.wav")
            sf.write(stem, normalize(voices[i] / 2**31), SAMPLE_RATE, subtype="PCM_16")
            print(f"  stem -> {stem}")
