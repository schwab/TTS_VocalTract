"""
Bit-faithful port of the Propeller VocalTract v1.1 synthesis engine.

This is a sample-by-sample re-implementation of the assembly cog in
``spin/VocalTract.spin``: same 32-bit integer arithmetic, same ROM tables,
same shift-and-add multiply, same 13-step CORDIC resonators, same LFSR noise,
and the same frame handler coroutine that interpolates between queued frames.
It is the project's ground truth for "what the Propeller would have played".

It is deliberately *not* differentiable and not part of the training path.
Uses:

- render the original Spin demos (see ``singing_demo.py``) to WAV
- generate target audio with known parameters (Phase 1 Step 7, Phase 3)
- validate ``params.py``'s unit conversions against the real engine

Engine summary (per 20 kHz sample, see the ASM for the exact order)::

    lfsr noise ──aa──────────────┐
    vibrato(vp,vr) + gp → antilog → glottal phase → antilog pulse ──ga──┤
                                                     x ─► F1 ─► F2 ─► F3 ─► F4 ─► nasal(na,nf) ─┬─► out
    lfsr-jittered sine at ff ──fa──────────────────────────────────────────────────────────────┘

Frames and timing
-----------------
A frame is the 13 parameter bytes plus a step size. ``go(time)`` on the
hardware computes ``stepsize = $01000000 / (time*100/pace #> 2)``; the engine
then advances every parameter by one interpolation step per 14 samples
(700 us), so ``time`` is in units of 700 us at pace 100. Each tract has an
8-slot queue, and a slot is only freed once its frame has *finished*
interpolating. The parent program blocks whenever the slot it wants is busy.

``render()`` co-simulates that: the queued frames for all tracts are fed in
program order, and the feed stalls (for every tract) whenever the next
frame's target queue is full -- exactly like the single Spin cog driving
four VocalTract objects in the singing demo. Spin interpreter execution time
is treated as zero (it is microseconds against 1.4 ms+ frames).
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .rom import antilog_table, sine_table

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without numba
    import warnings

    warnings.warn(
        "numba not installed: the VocalTract reference engine will run in pure "
        "Python, which is roughly 100x slower (minutes for the full singing demo)."
    )

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


SAMPLE_RATE = 20_000          # VocalTract's fixed sample rate
SAMPLES_PER_FRAME_STEP = 14   # 1 + 13 handler segments per interpolation step
FRAME_BUFFERS = 8             # queue depth per tract
PARAM_COUNT = 13

TUNE = 0x66920000             # 'tune': gp=100 -> 110.00 Hz (manually calibrated by Chip)
LFSR_TAPS = 0x80061000
CORDIC_DELTA = np.array([     # atan(2**-k) with 2**32 == 90 degrees, k = 1..12
    0x4B901476, 0x27ECE16D, 0x14444750, 0x0A2C350C,
    0x05175F85, 0x028BD879, 0x0145F154, 0x00A2F94D,
    0x00517CBB, 0x0028BE60, 0x00145F30, 0x000A2F98,
], dtype=np.int64)

ANTILOG = antilog_table()
SINE = sine_table()

M32 = 0xFFFFFFFF

# Parameter register indices (par_curr order in the ASM)
AA, GA, GP, VP, VR, F1, F2, F3, F4, NA, NF, FA, FF = range(PARAM_COUNT)

# Frame handler coroutine yield points (labels in the ASM)
_WAIT, _FINAL, _SET, _STEPFRAME, _STEP = range(5)


def stepsize(time: int, pace: int = 100) -> int:
    """Step size written by ``VocalTract.go(time)`` at the given pace."""
    return 0x01000000 // max(time * 100 // pace, 2)


@dataclass(frozen=True)
class QueuedFrame:
    """One ``go()`` call: the parameter bytes for a tract and their step size."""

    tract: int
    params: tuple[int, ...]   # 13 bytes, aa..ff
    step: int                 # from stepsize(time, pace)


# ---------------------------------------------------------------------------
# 32-bit register helpers. Registers are held as unsigned ints in [0, 2**32).
# ---------------------------------------------------------------------------

@njit(cache=True)
def _signed(v):
    v &= M32
    return v - 0x100000000 if v & 0x80000000 else v


@njit(cache=True)
def _sar(v, n):
    return (_signed(v) >> n) & M32


@njit(cache=True)
def _parity(v):
    v ^= v >> 16
    v ^= v >> 8
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    return v & 1


# ---------------------------------------------------------------------------
# Math subroutines (antilog / sine / mult / cordic in the ASM)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _antilog(t1, antilog):
    """t1 = log (4-bit whole, 11-bit fraction at the top) -> 2**log."""
    frac = antilog[(t1 >> 17) & 0x7FF]
    whole = (t1 >> 28) & 0xF
    return ((frac | 0x10000) << whole) & M32


@njit(cache=True)
def _mult(t1, t2):
    """15-bit unsigned (top of t1) x 17-bit signed (top of t2), 15 shift-add steps."""
    t1 = (t1 & M32) >> 17
    t2 = (_sar(t2, 15) << 14) & M32
    for _ in range(15):
        carry = t1 & 1
        t1 = _sar(t1, 1)
        if carry:
            t1 = (t1 + t2) & M32
    return t1


@njit(cache=True)
def _sine(t1, t2, sine):
    """Scale t1 (unsigned) times sin(t2) (13-bit angle at the top)."""
    t2 = (t2 & M32) >> 19
    if t2 & 0x800:              # quadrants 2|4 read the table backwards
        offset = (-t2) & 0xFFF
    else:
        offset = t2 & 0xFFF
    w = sine[offset]
    if t2 & 0x1000:             # quadrants 3|4 are negative
        w = -w
    return _mult(t1, (w << 15) & M32)


@njit(cache=True)
def _cordic(a, x, y, deltas):
    """Rotate (x, y) by a (2**32 == 90 degrees) after scaling by 0.60725*0.984.

    The 0.984 is the resonator damping: each formant is a complex one-pole
    filter with pole radius ~0.984 (about 100 Hz bandwidth at 20 kHz).
    """
    x = _sar(x, 1)
    x = (x + _sar(x, 3)) & M32
    x = (x + _sar(x, 4)) & M32
    y = _sar(y, 1)
    y = (y + _sar(y, 3)) & M32
    y = (y + _sar(y, 4)) & M32
    # first step (45 degrees) is inline in the ASM
    t1 = x
    x = (x - y) & M32
    y = (y + t1) & M32
    carry = a < 0x80000000      # 'sub a,h80000000 wc' borrow
    a = (a - 0x80000000) & M32
    for k in range(1, 14):
        if k > 1:
            carry = (a >> 31) != 0  # 'mov a,a wc'
        t1 = _sar(y, k)
        t2 = _sar(x, k)
        if carry:               # sumnc/sumc with C=1
            x = (x + t1) & M32
            y = (y - t2) & M32
            if k < 13:          # last 'sumnc a' is overwritten by 'ret'
                a = (a + deltas[k - 1]) & M32
        else:
            x = (x - t1) & M32
            y = (y + t2) & M32
            if k < 13:
                a = (a - deltas[k - 1]) & M32
    return x, y


# ---------------------------------------------------------------------------
# Co-simulation: parent feeding queues + N vocal tract cogs
# ---------------------------------------------------------------------------

@njit(cache=True)
def _render(op_tract, op_params, op_step, n_tracts, attenuation, max_samples,
            tail_samples, antilog, sine, deltas):
    out = np.zeros((n_tracts, max_samples), dtype=np.int32)

    # per-tract queues (written by the parent, consumed by the engine)
    slot_step = np.zeros((n_tracts, FRAME_BUFFERS), dtype=np.int64)
    slot_par = np.zeros((n_tracts, FRAME_BUFFERS, PARAM_COUNT), dtype=np.int64)
    write_index = np.zeros(n_tracts, dtype=np.int64)

    # engine registers
    par_curr = np.zeros((n_tracts, PARAM_COUNT), dtype=np.int64)
    par_next = np.zeros((n_tracts, PARAM_COUNT), dtype=np.int64)
    par_step = np.zeros((n_tracts, PARAM_COUNT), dtype=np.int64)
    fx = np.zeros((n_tracts, 4), dtype=np.int64)
    fy = np.zeros((n_tracts, 4), dtype=np.int64)
    lfsr = np.ones(n_tracts, dtype=np.int64)
    x = np.zeros(n_tracts, dtype=np.int64)
    nx = np.zeros(n_tracts, dtype=np.int64)
    vphase = np.zeros(n_tracts, dtype=np.int64)
    gphase = np.zeros(n_tracts, dtype=np.int64)
    fphase = np.zeros(n_tracts, dtype=np.int64)

    # frame handler coroutine state
    cont = np.zeros(n_tracts, dtype=np.int64)          # _WAIT
    frame_index = np.zeros(n_tracts, dtype=np.int64)
    frame_cnt = np.zeros(n_tracts, dtype=np.int64)
    par_i = np.zeros(n_tracts, dtype=np.int64)
    step_size = np.zeros(n_tracts, dtype=np.int64)
    step_acc = np.zeros(n_tracts, dtype=np.int64)

    n_ops = op_tract.shape[0]
    next_op = 0
    end_sample = -1

    for n in range(max_samples):
        # parent: queue frames in program order until a target queue is full
        while next_op < n_ops:
            tr = op_tract[next_op]
            wi = write_index[tr]
            if slot_step[tr, wi] != 0:
                break
            for p in range(PARAM_COUNT):
                slot_par[tr, wi, p] = op_params[next_op, p]
            slot_step[tr, wi] = op_step[next_op]
            write_index[tr] = (wi + 1) % FRAME_BUFFERS
            next_op += 1

        if end_sample < 0 and next_op == n_ops and not slot_step.any():
            end_sample = n
        if end_sample >= 0 and n >= end_sample + tail_samples:
            return out[:, :n]

        for tr in range(n_tracts):
            pc = par_curr[tr]

            # output previous sample, after master attenuation
            out[tr, n] = _signed(_sar(x[tr], attenuation[tr]))

            # white noise source: iterate lfsr three times
            r = lfsr[tr]
            for _ in range(3):
                r = ((r << 1) | _parity(r & LFSR_TAPS)) & M32
            lfsr[tr] = r

            # aspiration
            xv = _sar(_mult(pc[AA], r), 8)

            # vibrato
            vphase[tr] = (vphase[tr] + (pc[VR] >> 10)) & M32
            t1 = (_sine(pc[VP], vphase[tr], sine) + pc[GP]) & M32

            # glottal pulse: pitch/3 meshes 1/48-octave gp with 2**28 per octave
            t1 >>= 2
            t1 = (t1 + (t1 >> 2)) & M32
            t1 = (t1 + (t1 >> 4)) & M32
            t1 = (t1 + (t1 >> 8)) & M32
            t1 = (t1 + TUNE) & M32
            gphase[tr] = (gphase[tr] + _antilog(t1, antilog)) & M32
            t2 = (_antilog(gphase[tr], antilog) - 0x40000000) & M32
            xv = (xv + _sar(_sine(pc[GA], t2, sine), 6)) & M32

            # vocal tract formants: cascade of 4 complex resonators
            yv = 0
            for f in range(4):
                xv = (xv + fx[tr, f]) & M32
                yv = (yv + fy[tr, f]) & M32
                xv, yv = _cordic(pc[F1 + f], xv, yv, deltas)
                fx[tr, f] = xv
                fy[tr, f] = yv

            # nasal anti-formant
            nx[tr] = (nx[tr] + xv) & M32
            xr, yv = _cordic(pc[NF], xv, yv, deltas)
            t1 = _mult(pc[NA], xr)
            xv = nx[tr]
            nx[tr] = (-t1) & M32

            # frication: sine at ff with lfsr phase jitter
            t1 = _sar(r, 3)
            fp = (fphase[tr] + t1) & M32
            t1 = _sar(t1, 1)
            fp = (fp + t1) & M32
            fp = (fp + (pc[FF] >> 1)) & M32
            fphase[tr] = fp
            xv = (xv + _sine(pc[FA], fp, sine)) & M32
            x[tr] = xv

            # frame handler: run one coroutine segment per sample
            st = cont[tr]
            fi = frame_index[tr]
            i = par_i[tr]
            if st == _WAIT:
                ss = slot_step[tr, fi] & 0x00FFFFFF
                frame_cnt[tr] = PARAM_COUNT
                par_i[tr] = 0
                if ss == 0:
                    cont[tr] = _FINAL
                else:
                    step_size[tr] = ss + 1
                    step_acc[tr] = ss + 1
                    cont[tr] = _SET
            elif st == _FINAL:
                pc[i] = par_next[tr, i]
                par_i[tr] = i + 1
                frame_cnt[tr] -= 1
                cont[tr] = _FINAL if frame_cnt[tr] else _WAIT
            elif st == _SET:
                new = slot_par[tr, fi, i] << 24
                old = par_next[tr, i]
                pc[i] = old
                par_next[tr, i] = new
                negative = new < old
                d = (old - new) if negative else (new - old)
                for _ in range(8):      # delta * step_size, msb first
                    carry = (d >> 31) & 1
                    d = (d << 1) & M32
                    if carry:
                        d = (d + step_size[tr]) & M32
                par_step[tr, i] = ((-d) & M32) if negative else d
                par_i[tr] = i + 1
                frame_cnt[tr] -= 1
                cont[tr] = _SET if frame_cnt[tr] else _STEPFRAME
            elif st == _STEPFRAME:
                par_i[tr] = 0
                frame_cnt[tr] = PARAM_COUNT
                cont[tr] = _STEP
            else:  # _STEP
                pc[i] = (pc[i] + par_step[tr, i]) & M32
                par_i[tr] = i + 1
                frame_cnt[tr] -= 1
                if frame_cnt[tr]:
                    cont[tr] = _STEP
                else:
                    step_acc[tr] = (step_acc[tr] + step_size[tr]) & M32
                    if step_acc[tr] & 0x01000000:   # frame done: free the slot
                        slot_step[tr, fi] = 0
                        frame_index[tr] = (fi + 1) % FRAME_BUFFERS
                        cont[tr] = _WAIT
                    else:
                        cont[tr] = _STEPFRAME

    return out


def render(frames: Sequence[QueuedFrame], n_tracts: int | None = None,
           attenuation: Sequence[int] | int = 0, tail_seconds: float = 0.25) -> np.ndarray:
    """Render queued frames for one or more vocal tracts.

    Args:
        frames: ``go()`` calls in program order (any mix of tracts).
        n_tracts: number of VocalTract instances (default: max tract + 1).
        attenuation: master attenuation 0..7 (3 dB steps), as set_attenuation();
            one value for all tracts or one per tract.
        tail_seconds: audio to keep after the last frame has finished.

    Returns:
        ``(n_tracts, samples)`` int32 array of raw engine samples at
        ``SAMPLE_RATE`` -- the signed 32-bit values VocalTract writes to its
        ``sample`` long. Divide by 2**31 for floats in [-1, 1).
    """
    if not frames:
        raise ValueError("no frames to render")
    if n_tracts is None:
        n_tracts = max(f.tract for f in frames) + 1
    op_tract = np.array([f.tract for f in frames], dtype=np.int64)
    op_params = np.array([f.params for f in frames], dtype=np.int64)
    op_step = np.array([f.step for f in frames], dtype=np.int64)
    if op_params.shape[1:] != (PARAM_COUNT,) or op_params.min() < 0 or op_params.max() > 255:
        raise ValueError("each frame needs 13 parameter bytes in 0..255")
    if op_step.min() < 1 or op_step.max() > 0x00FFFFFF:
        raise ValueError("frame step sizes must be in 1..$FFFFFF")

    att = np.broadcast_to(np.asarray(attenuation, dtype=np.int64), (n_tracts,)).copy()
    if att.min() < 0 or att.max() > 7:
        raise ValueError("attenuation must be 0..7")
    tail = int(tail_seconds * SAMPLE_RATE)
    # Upper bound: every frame interpolated back to back on a single tract.
    steps = (0x01000000 // op_step) + 2
    max_samples = int(steps.sum()) * SAMPLES_PER_FRAME_STEP + tail + 64
    return _render(op_tract, op_params, op_step, n_tracts, att,
                   max_samples, tail, ANTILOG, SINE, CORDIC_DELTA)


def render_tract(frames: Sequence[tuple[Sequence[int], int]], pace: int = 100,
                 **kwargs) -> np.ndarray:
    """Convenience wrapper for a single tract: ``[(params13, time), ...]`` -> samples."""
    queued = [QueuedFrame(0, tuple(int(v) for v in p), stepsize(t, pace)) for p, t in frames]
    return render(queued, n_tracts=1, **kwargs)[0]
