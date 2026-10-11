"""
Render Chip Gracey's "Singing Demo" (vowels) with the reference VocalTract engine.

``spin/SingingDemo.spin`` sings the same 16th-century four-part score as the
"Seven" demo, but each voice holds one vowel for the whole song (basso "oo",
tenor "ah", alto "ee", soprano "aw"-ish) with legato note changes, a slow
vibrato, and slightly randomized note lengths from Spin's ``?rnd`` operator
(reproduced exactly, so every render is identical -- as on the chip, where
``rnd`` starts at 0).

Each note is only 2-4 frames, so the 8-frame queues never stall the single
feeding cog for long and the faithful render stays in lockstep; ``--sync``
makes little difference here.

Usage::

    python -m vocaltract.reference.singing_demo            # -> out/singing_demo.wav
    python -m vocaltract.reference.singing_demo --stems    # also one WAV per voice
"""

from pathlib import Path

from .engine import F1, GA, GP, VP, VR, QueuedFrame
from .spin import (SPIN_DIR, DemoRender, SpinRandom, SpinSource, VocalTractProgram, parse_spin,
                   run_demo_cli)

DEFAULT_SPIN = SPIN_DIR / "SingingDemo.spin"
DEFAULT_OUT = Path("out") / "singing_demo.wav"


class SingingDemo(VocalTractProgram):
    """Port of SingingDemo.spin's PUB/PRI methods."""

    def __init__(self, src: SpinSource):
        super().__init__(4)
        self.const = src.constants
        self.parts = [src.notes(label) for label in src.tables["parts"]]
        self.volume = src.tables["volume"]
        self.state = [0] * 4
        self.rnd = SpinRandom(0)        # 'long rnd' (VAR longs start at 0)
        self.angles = [0] * 4

    def set_formants(self, tract: int, sf1: int, sf2: int, sf3: int, sf4: int) -> None:
        for i, sf in enumerate((sf1, sf2, sf3, sf4)):
            self.set(tract, F1 + i, sf // 19)

    def run(self) -> list[QueuedFrame]:
        for i in range(4):
            self.set_attenuation(i, 1)
            self.angles[i] = i * 0x5555             # spread across the sound stage

        self.set_formants(0, 275, 850, 2400, 3520)  # each tract a different vowel
        self.set_formants(1, 700, 1800, 2550, 3400)
        self.set_formants(2, 280, 2040, 3040, 3600)
        self.set_formants(3, 650, 1200, 2500, 3500)

        for i in range(4):
            self.set(i, VP, 4)                       # vibrato pitch range
            self.set(i, VR, (66, 68, 71, 75)[i])     # lookupz: different vibrato rates
            self.go(i, 1000 + (self.rnd.next() >> 24))  # slide to formants while silent
            self.set_pace(i, 100)

        while any(s != -1 for s in self.state):
            for tract in range(4):                   # basso, tenor, alto, soprano
                self.advance(tract)
        return self.frames

    def advance(self, tract: int) -> None:
        if self.state[tract] == -1:
            return
        vol = self.volume[tract]
        part = self.parts[tract]
        note = part[self.state[tract]] if self.state[tract] < len(part) else 0
        self.state[tract] += 1
        if note:
            self.set(tract, GP, (note & 0xFC) + self.const["shift"])
            if note & self.const["pausenote"]:
                self.set(tract, GA, vol)
                self.go(tract, 200)
                self.go(tract, 1000)
                self.set(tract, GA, 0)
                self.go(tract, 500)
                self.go(tract, 750)
            elif note & self.const["halfnote"]:
                self.set(tract, GA, vol)
                self.go(tract, 200)
                self.go(tract, 50)
                self.go(tract, 200)
                self.go(tract, 50)
            else:                                    # normal
                time = 200
                if not self.get(tract, GA):          # coming out of a rest: jump pitch silently
                    self.go(tract, 2)
                    time = 500
                self.set(tract, GA, vol)
                self.go(tract, time)
                self.go(tract, 1000 - time + (self.rnd.next() >> 27))
        else:                                        # end of notes
            self.set(tract, GA, 0)
            self.go(tract, 200)
            self.go(tract, 2000)
            self.state[tract] = -1


def build(spin_path: Path = DEFAULT_SPIN) -> DemoRender:
    src = parse_spin(spin_path)
    demo = SingingDemo(src)
    frames = demo.run()
    return DemoRender(frames=frames, voice_names=list(src.tables["parts"]),
                      angles=demo.angles, attenuation=demo.attenuation)


def main(argv: list[str] | None = None) -> None:
    run_demo_cli(__doc__.strip().split("\n")[0], DEFAULT_SPIN, DEFAULT_OUT, build, argv)


if __name__ == "__main__":
    main()
