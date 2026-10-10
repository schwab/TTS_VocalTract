"""
Propeller P1 hub-ROM math tables used by VocalTract.spin.

VocalTract's assembly reads two lookup tables out of the chip's built-in ROM:

- the **antilog table** at ``$D000``: 2048 words, fractional part of
  ``2**(i/2048)`` scaled to 16 bits
- the **sine table** at ``$E000``: 2049 words, the first quadrant of a sine
  wave (0..90 degrees inclusive) scaled to 0..$FFFF

Rather than ship Parallax's ROM image, the tables are regenerated from their
defining formulas. ``tests/test_reference.py`` checks these against the real
ROM image (Gear's ``StartupROM.bin``) when one is available: both match
bit-for-bit.
"""

import numpy as np

ANTILOG_ENTRIES = 2048
SINE_ENTRIES = 2049  # 0..90 degrees inclusive


def antilog_table() -> np.ndarray:
    """ROM $D000: round(65536 * (2**(i/2048) - 1)) for i in 0..2047."""
    i = np.arange(ANTILOG_ENTRIES)
    return np.round(65536.0 * (2.0 ** (i / ANTILOG_ENTRIES) - 1.0)).astype(np.int64)


def sine_table() -> np.ndarray:
    """ROM $E000: round(65535 * sin(i/2048 * pi/2)) for i in 0..2048."""
    i = np.arange(SINE_ENTRIES)
    return np.round(65535.0 * np.sin(i / 2048.0 * np.pi / 2.0)).astype(np.int64)


def tables_from_rom_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the antilog and sine tables from a 32 KB ROM image ($8000..$FFFF)."""
    with open(path, "rb") as f:
        rom = np.frombuffer(f.read(), dtype="<u2").astype(np.int64)
    antilog_start = (0xD000 - 0x8000) // 2
    sine_start = (0xE000 - 0x8000) // 2
    return (
        rom[antilog_start:antilog_start + ANTILOG_ENTRIES],
        rom[sine_start:sine_start + SINE_ENTRIES],
    )
