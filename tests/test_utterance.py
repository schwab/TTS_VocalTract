"""Tests for Phase 2 utterance fitting (vocaltract.phase2)."""

import numpy as np
import pytest
import torch

from vocaltract.analysis.resynth import HOP_S as ENGINE_HOP_S
from vocaltract.analysis.resynth import render_frames
from vocaltract.phase2.harness import corpus, report
from vocaltract.phase2.utterance import (
    FitConfig,
    Utterance,
    engine_scale,
    engine_to_grid,
    fit,
    frame_levels_db,
    prepare,
    render_engine,
    split_sources,
)
from vocaltract.reference.engine import AA, F1, FA, FF, GA, GP

HOP = 160


def vowel_frames(n: int = 40, ga: int = 80, aa: int = 0, fa: int = 0) -> np.ndarray:
    """Level-A style uint8 frames: a steady /a/ at ~120 Hz."""
    frames = np.zeros((n, 13), dtype=np.uint8)
    frames[:, GA], frames[:, AA], frames[:, GP] = ga, aa, 106
    frames[:, F1:F1 + 4] = (37, 56, 125, 174)
    frames[:, FA], frames[:, FF] = fa, 120
    return frames


def n_samples(frames: np.ndarray) -> int:
    return int(len(frames) * ENGINE_HOP_S * 16000) // HOP * HOP


class TestGrid:
    def test_ramps_from_zero_then_holds(self):
        frames = np.full((10, 13), 255, dtype=np.uint8)
        grid = engine_to_grid(frames, n_frames=9, hop=HOP)
        assert grid.shape == (9, 13)
        # first centre (5 ms) is halfway up the ramp to frame 0 at 9.8 ms
        np.testing.assert_allclose(grid[0], 5.0 / 9.8, rtol=1e-6)
        np.testing.assert_allclose(grid[2:], 1.0)

    def test_follows_linear_interpolation_between_frames(self):
        frames = np.zeros((3, 13))
        frames[:, 0] = (0, 100, 200)  # reached at 9.8, 19.6, 29.4 ms
        grid = engine_to_grid(frames, n_frames=3, hop=HOP)  # centres 5, 15, 25 ms
        expected = np.interp([0.005, 0.015, 0.025], [0, 0.0098, 0.0196, 0.0294], [0, 0, 100, 200]) / 255
        np.testing.assert_allclose(grid[:, 0], expected)


class TestSources:
    def test_split_renders_add_up_to_full_render(self):
        """The engine is linear in its sources: voice + frication = full render (mod 2**32)."""
        frames = vowel_frames(ga=30, aa=10, fa=15)
        voice, fric = split_sources(frames)
        assert not voice[:, FA].any() and not fric[:, [GA, AA]].any()
        parts = render_frames(voice).astype(np.int64) + render_frames(fric).astype(np.int64)
        assert not np.any((parts - render_frames(frames).astype(np.int64)) % 2**32)

    def test_split_renders_add_up_without_overflow(self):
        frames = vowel_frames(ga=8, aa=3, fa=4)
        voice, fric = split_sources(frames)
        n = n_samples(frames)
        np.testing.assert_allclose(render_engine(voice, n) + render_engine(fric, n),
                                   render_engine(frames, n), atol=1e-12)

    def test_engine_scale_recovers_a_gain(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(16000)
        assert engine_scale(3.0 * x, x, HOP) == pytest.approx(3.0, rel=1e-6)

    @pytest.mark.parametrize("source", ["ga", "aa"])
    def test_unnormalized_pipeline_tracks_engine_level(self, source):
        """
        Same frames, engine vs pipeline: one constant scale for both sources
        (aspiration calibrated to the engine's, see source.ENGINE_NOISE_RMS).
        """
        reference = vowel_frames(ga=80, aa=0)
        frames = vowel_frames(ga=80 if source == "ga" else 0, aa=80 if source == "aa" else 0)
        n = n_samples(frames)
        model = Utterance(None, FitConfig(hop=HOP))

        def pipeline(f):
            return model.render(torch.from_numpy(engine_to_grid(f, n // HOP, HOP)).float()[None])

        scale = engine_scale(render_engine(reference, n), pipeline(reference), HOP)
        engine, pipe = render_engine(frames, n), scale * pipeline(frames)
        skip = 5  # onset ramp
        diff = frame_levels_db(pipe, HOP)[skip:] - frame_levels_db(engine, HOP)[skip:]
        assert abs(np.median(diff)) < 1.5


class TestFit:
    def test_prepare_mixes_like_level_a(self):
        frames = vowel_frames(ga=60, aa=10, fa=40)
        n = n_samples(frames)
        init, frication, level_a = prepare(frames, n, HOP)
        assert init.shape == (n // HOP, 13) and frication.shape == (n,) and level_a.shape == (n,)
        assert np.abs(frication).max() > 0

    def test_loss_decreases_and_shapes(self):
        frames = vowel_frames(n=30, ga=60, fa=20)
        n = n_samples(frames)
        init, frication, level_a = prepare(frames, n, HOP)
        start = init.copy()
        start[:, F1:F1 + 4] += 0.03  # move the formants off the target
        result = fit(level_a, start, frication, FitConfig(hop=HOP, steps=15, log_every=5))
        assert result.history[result.best_step] < result.history[0]
        assert result.params.shape == (1, len(init), 13)
        assert result.audio.shape == (len(init) * HOP,)
        assert len(result.bandwidths_hz) == 4 and result.frication_gain > 0
        assert result.model.render(result.params, seed=3).shape == (len(init) * HOP,)

    def test_rejects_short_target(self):
        with pytest.raises(ValueError):
            fit(np.zeros(100), np.zeros((5, 13)), None, FitConfig(hop=HOP, steps=0))


def record(clip_id, wers, words=10):
    metrics = {cond: {"wer": w, "errors": w * words, "words": words}
               for cond, w in zip(("original", "level_a", "init", "fit"), wers)}
    return {"id": clip_id, "gender": "F", "median_f0_hz": 200.0, "metrics": metrics,
            "config": {"hop": 160, "steps": 1, "lr": 1e-3}, "asr": "small.en", "beam": 5, "seeds": [1]}


class TestReport:
    def test_corpus_wer_weights_by_words(self):
        records = [record("a", (0, 0.5, 0.5, 0.2), words=10), record("b", (0, 0.1, 0.1, 0.1), words=30)]
        summary = corpus(records, "level_a")
        assert summary["wer"] == pytest.approx((5 + 3) / 40)
        assert summary["worst_id"] == "a" and summary["median"] == pytest.approx(0.3)

    def test_acceptance_fails_when_one_clip_is_worse(self):
        text = report([record("a", (0, 0.5, 0.5, 0.2)), record("b", (0, 0.1, 0.1, 0.2))])
        assert "clips worse than their level A: b" in text and "FAIL" in text

    def test_acceptance_passes(self):
        assert "PASS" in report([record("a", (0, 0.5, 0.5, 0.2))])
