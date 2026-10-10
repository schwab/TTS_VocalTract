"""
Render Chip Gracey's "Singing Demo (Seven)" with the reference VocalTract engine.

``spin/SingingDemoSeven.spin`` runs four VocalTract objects that sing a
16th-century four-part harmony, every note on the word "seven". The score
data is read from the .spin file; the Spin methods are ported line by line
below.

Timing: one Spin cog feeds all four tracts round-robin, a whole "seven"
(18 frames) per ``advance()``, but each VocalTract queue holds only 8 frames
and frees a slot only when that frame *finishes*. So every ``advance()``
blocks for about half a note, and the faithful render has the voices entering
~0.5 s apart, each singing once every ~2 s (~76 s total). ``--sync`` instead
feeds each voice independently, which keeps the parts in lockstep as written
(~41 s).

Usage::

    python -m vocaltract.reference.singing_demo_seven            # -> out/singing_demo_seven.wav
    python -m vocaltract.reference.singing_demo_seven --stems    # also one WAV per voice
    python -m vocaltract.reference.singing_demo_seven --sync     # parts in lockstep
"""

from pathlib import Path

from .engine import AA, FA, FF, F1, GA, GP, NA, NF, VP, VR, QueuedFrame
from .spin import SPIN_DIR, DemoRender, SpinSource, VocalTractProgram, parse_spin, run_demo_cli

DEFAULT_SPIN = SPIN_DIR / "SingingDemoSeven.spin"
DEFAULT_OUT = Path("out") / "singing_demo_seven.wav"


class SingingDemoSeven(VocalTractProgram):
    """Port of SingingDemoSeven.spin's PUB/PRI methods."""

    def __init__(self, src: SpinSource):
        super().__init__(4)
        self.src = src
        self.const = src.constants
        self.parts = [src.notes(label) for label in src.tables["parts"]]
        self.lowvol = src.tables["lowvol"]
        self.hivol = src.tables["hivol"]
        self.state = [0] * 4
        self.tract = 0              # 'long tract' -- the demo's implicit current tract

    # The demo's PRI set/go/set_formants act on the current 'tract'.
    def tset(self, parameter: int, value: int) -> None:
        self.set(self.tract, parameter, value)

    def tgo(self, time: int) -> None:
        self.go(self.tract, time)

    def set_formants(self, sf1: int, sf2: int, sf3: int, sf4: int) -> None:
        for i, sf in enumerate((sf1, sf2, sf3, sf4)):
            self.tset(F1 + i, sf // (19 - self.tract))

    def run(self) -> list[QueuedFrame]:
        for self.tract in range(4):
            self.tset(VP, 1)                       # vibrato pitch range
            self.tset(VR, 55 + (self.tract << 2))  # slightly different vibrato rates
        while any(s != -1 for s in self.state):
            for tract in range(4):                 # basso, tenor, alto, soprano
                self.advance(tract)
        return self.frames

    def advance(self, tractx: int) -> None:
        self.tract = t = tractx
        if self.state[t] == -1:
            return
        part = self.parts[t]
        note = part[self.state[t]] if self.state[t] < len(part) else 0
        self.state[t] += 1
        if note:
            self.tset(GP, (note & 0xFC) + self.const["shift"])
            if note & self.const["pausenote"]:
                self.seven(50)
                self.tgo(1000)
            elif note & self.const["halfnote"]:
                self.seven(200)
            else:
                self.seven(100)
        else:
            self.state[t] = -1                     # end of notes

    def seven(self, pace: int) -> None:
        t, hivol, lowvol = self.tract, self.hivol, self.lowvol
        self.set_pace(t, pace)

        self.set_formants(470, 1650, 2500, 3500)
        self.tset(FF, 165 + (t << 3))
        self.tgo(10)
        self.tset(FA, hivol[t] // 3)               # 's'
        self.tgo(200)
        self.tset(AA, hivol[t] // 2)
        self.tgo(50)
        self.tset(FA, 0)
        self.tgo(50)
        self.tset(GA, hivol[t])                    # 'e'
        self.set_formants(700, 1750, 2500, 3500)
        self.tgo(70)
        self.set_formants(700, 1500, 2400, 3400)
        self.tgo(150)
        self.set_formants(600, 1440, 2300, 3300)
        self.tgo(50)
        self.tset(GA, lowvol[t])                   # 'v'
        self.tset(AA, 0)
        self.tset(FF, 240 + (t << 2))
        self.tgo(20)
        self.tset(FA, hivol[t] // 4)
        self.tgo(20)
        self.tgo(80)
        self.tset(FA, 0)
        self.tgo(50)
        self.tset(GA, hivol[t])                    # 'e'
        self.tset(AA, hivol[t] >> 2)
        self.set_formants(500, 1440, 2300, 3300)
        self.tgo(25)
        self.set_formants(550, 1750, 2400, 3400)
        self.tgo(60)
        self.tgo(50)
        self.set_formants(250, 1700, 2300, 3400)  # 'n'
        self.tset(NF, 2000 // (19 - t))
        self.tset(NA, 0xFF)
        self.tgo(60)
        self.tset(GA, hivol[t])
        self.tgo(150)
        self.tset(GA, 0)
        self.tset(AA, 0)
        self.tgo(80)
        self.tset(NA, 0)
        self.tgo(200)


def build(spin_path: Path = DEFAULT_SPIN) -> DemoRender:
    src = parse_spin(spin_path)
    demo = SingingDemoSeven(src)
    return DemoRender(frames=demo.run(), voice_names=list(src.tables["parts"]),
                      angles=src.tables["angles"], attenuation=demo.attenuation)


def main(argv: list[str] | None = None) -> None:
    run_demo_cli(__doc__.strip().split("\n")[0], DEFAULT_SPIN, DEFAULT_OUT, build, argv)


if __name__ == "__main__":
    main()
