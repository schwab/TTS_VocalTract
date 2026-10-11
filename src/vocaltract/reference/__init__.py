"""
Reference (non-differentiable) port of the Propeller VocalTract engine.

Ground truth for what the original hardware synthesizer produces from a
sequence of 13-byte parameter frames. See ``engine.py`` for the engine and
``singing_demo.py`` for a full port of Chip Gracey's Singing Demo (Seven).
"""

from .engine import SAMPLE_RATE, QueuedFrame, render, render_tract, stepsize

__all__ = ["SAMPLE_RATE", "QueuedFrame", "render", "render_tract", "stepsize"]
