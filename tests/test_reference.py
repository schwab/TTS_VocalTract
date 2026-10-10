"""Tests for the reference VocalTract engine port and the singing demo score."""

import os
from pathlib import Path

import numpy as np
import pytest

from vocaltract.reference import engine
from vocaltract.reference.engine import SAMPLE_RATE, render_tract, stepsize
from vocaltract.reference.rom import antilog_table, sine_table, tables_from_rom_image
from vocaltract.reference import singing_demo, singing_demo_seven
from vocaltract.reference.spin import (SPIN_DIR, SpinRandom, eval_expr, parse_spin,
                                       random_forward, render_voices)

GEAR_ROM = Path(os.environ.get(
    "GEAR_ROM", Path.home() / "project/gear-emu/Gear/Resources/StartupROM.bin"))

# aa ga  gp vp vr f1  f2  f3   f4  na nf fa ff
VOWEL = [0, 60, 100, 0, 0, 36, 56, 128, 179, 0, 0, 0, 0]


def steady(params, seconds=1.2):
    """Render a held frame and drop the first 0.2 s of onset."""
    y = render_tract([(params, 2), (params, int(seconds / 0.0007))], tail_seconds=0)
    return y[int(0.2 * SAMPLE_RATE):].astype(np.float64)


def f0(y):
    y = y - y.mean()
    r = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * len(y))) ** 2)
    lag = np.argmax(r[40:1000]) + 40
    return SAMPLE_RATE / lag


class TestRomTables:
    @pytest.mark.skipif(not GEAR_ROM.exists(), reason="no Propeller ROM image (set GEAR_ROM)")
    def test_tables_match_rom_image(self):
        antilog, sine = tables_from_rom_image(str(GEAR_ROM))
        np.testing.assert_array_equal(antilog_table(), antilog)
        np.testing.assert_array_equal(sine_table(), sine)

    def test_table_endpoints(self):
        assert antilog_table()[0] == 0
        assert sine_table()[0] == 0 and sine_table()[-1] == 0xFFFF


class TestMath:
    def test_sine_full_scale_range(self):
        """Full-scale sine is the exact 15-bit x 17-bit product $7FFF * $FFFF.

        The ASM comment quotes "$80014000..$7FFEC000", which is a rounded
        figure (not an exact product of any 15x17-bit pair); the shift-add
        multiply itself is exact here.
        """
        peak = engine._sine(0xFFFFFFFF, 0x40000000, engine.SINE)   # 90 degrees
        trough = engine._sine(0xFFFFFFFF, 0xC0000000, engine.SINE)  # 270 degrees
        assert peak == 0x7FFF * 0xFFFF
        assert trough == (-0x7FFF * 0xFFFF) & 0xFFFFFFFF

    def test_antilog(self):
        assert engine._antilog(0, engine.ANTILOG) == 0x10000
        assert engine._antilog(1 << 28, engine.ANTILOG) == 0x20000   # +1 octave doubles

    def test_stepsize_matches_go(self):
        assert stepsize(10) == 0x01000000 // 10
        assert stepsize(1000, pace=50) == 0x01000000 // 2000
        assert stepsize(0) == 0x01000000 // 2    # '#> 2' floor


class TestEngine:
    def test_silence_without_sources(self):
        quiet = [0, 0, 100, 0, 0, 36, 56, 128, 179, 0, 0, 0, 0]
        assert not render_tract([(quiet, 100)]).any()

    @pytest.mark.parametrize("gp, hz", [(100, 110.0), (148, 220.0), (52, 55.0)])
    def test_pitch_calibration(self, gp, hz):
        """Header table: gp=100 -> 110.00 Hz, 48 units per octave."""
        p = list(VOWEL)
        p[2] = gp
        assert f0(steady(p)) == pytest.approx(hz, rel=0.01)

    def test_formants(self):
        """Header table: f1..f4 in 19.53 Hz units; peaks of the harmonic envelope."""
        y = steady(VOWEL)
        spec = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(len(y))))
        freqs = np.fft.rfftfreq(len(y), 1 / SAMPLE_RATE)
        harmonics = 110.0 * np.arange(1, 45)
        amp = np.array([spec[np.abs(freqs - h) < 20].max() for h in harmonics])
        peaks = [harmonics[i] for i in range(1, len(amp) - 1)
                 if amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]]
        for formant_raw in (36, 56, 128, 179):
            target = formant_raw * SAMPLE_RATE / 1024
            assert min(abs(p - target) for p in peaks) < 60   # harmonic spacing 110 Hz

    def test_frame_timing(self):
        """go(time) lasts time * 700 us; frames play back to back."""
        loud = [0, 60, 100, 0, 0, 36, 56, 128, 179, 0, 0, 0, 0]
        y = render_tract([(loud, 1000)], tail_seconds=0)
        assert len(y) / SAMPLE_RATE == pytest.approx(0.7, abs=0.002)


class TestSpinHelpers:
    def test_random_forward(self):
        """?var treats 0 as 1, never returns 0, and is deterministic."""
        a = random_forward(0)
        assert a != 0 and a == random_forward(1)
        assert random_forward(a) != a

    def test_spin_random_is_signed(self):
        r = SpinRandom(0)
        vals = [r.next() >> 24 for _ in range(200)]
        assert min(vals) >= -128 and max(vals) <= 127
        assert min(vals) < 0 < max(vals)

    def test_eval_expr(self):
        names = {"o2G": 140, "pausenote": 1}
        assert eval_expr("o2G + pausenote", names) == 141
        assert eval_expr("170-35", names) == 135
        assert eval_expr("$5C00", names) == 0x5C00
        assert eval_expr("%000_110", names) == 6
        assert eval_expr("-24", names) == -24

    @pytest.mark.parametrize("name", ["SingingDemoSeven.spin", "SingingDemo.spin"])
    def test_parse_demo_sources(self, name):
        """UTF-8 and UTF-16 (Propeller Tool) files both parse to the same score."""
        src = parse_spin(SPIN_DIR / name)
        assert src.constants["shift"] == -24
        assert (src.constants["pausenote"], src.constants["halfnote"]) == (1, 2)
        assert src.constants["o2G"] == 140          # 2*48 + 44
        assert src.constants["ff"] == 12            # '#0, aa,ga,...,ff'
        assert src.tables["parts"] == ["basso", "tenor", "alto", "soprano"]
        assert [len(src.notes(p)) for p in src.tables["parts"]] == [34, 33, 32, 32]

    def test_sync_render_keeps_each_voice_length(self):
        loud = (0, 60, 100, 0, 0, 36, 56, 128, 179, 0, 0, 0, 0)
        from vocaltract.reference.engine import QueuedFrame
        frames = [QueuedFrame(0, loud, stepsize(100)), QueuedFrame(1, loud, stepsize(300))]
        v = render_voices(frames, 2, sync=True)
        assert v.shape[0] == 2 and not v[0, -100:].any()  # voice 0 ended, padded


class TestSingingDemoSeven:
    def test_score(self):
        src = parse_spin(singing_demo_seven.DEFAULT_SPIN)
        assert src.tables["hivol"] == [70, 60, 50, 40]
        assert src.tables["lowvol"] == [10, 8, 6, 5]
        assert src.tables["angles"] == [0x0000, 0x5C00, 0xA400, 0xFFFF]

    def test_frames(self):
        frames = singing_demo_seven.build().frames
        assert len(frames) == 2373        # 131 notes * 18 + 15 pause rests
        first = frames[0]
        assert first.tract == 0
        # gp = o2G + shift(-24); vp=1, vr=55; formants 470,1650,2500,3500 / 19; ff=165
        assert first.params == (0, 0, 116, 1, 55, 24, 86, 131, 184, 0, 0, 0, 165)
        assert first.step == stepsize(10)


class TestSingingDemo:
    def test_frames(self):
        demo = singing_demo.build()
        assert demo.attenuation == [1, 1, 1, 1]
        assert demo.angles == [0, 0x5555, 0xAAAA, 0xFFFF]
        first = demo.frames[:4]
        # silent slide to each tract's vowel: vp=4, vr=lookupz(66,68,71,75), formants/19
        assert [f.params[3:9] for f in first] == [
            (4, 66, 14, 44, 126, 185), (4, 68, 36, 94, 134, 178),
            (4, 71, 14, 107, 160, 189), (4, 75, 34, 63, 131, 184)]
        assert all(f.params[1] == 0 for f in first)     # ga = 0
        # first note: jump pitch silently (go 2), then voice in over 500
        assert demo.frames[4].step == stepsize(2)
        assert demo.frames[5].params[1] == 40 and demo.frames[5].step == stepsize(500)

    def test_deterministic(self):
        assert singing_demo.build().frames == singing_demo.build().frames
