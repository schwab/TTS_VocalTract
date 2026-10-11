"""Tests for the Phase 1 single-clip overfit demo."""

import subprocess
import sys

import pytest
import soundfile as sf
import torch

from vocaltract.phase1 import overfit_demo as demo


@pytest.fixture
def out_path(tmp_path):
    return tmp_path / "overfit.wav"


class TestAssets:
    def test_manifest_lists_all_targets(self):
        names = demo.list_targets()
        assert names[0] == "synthetic_vowel_ah"
        assert {"vt_vowel_ah", "speech_male", "speech_female"} <= set(names)

    @pytest.mark.parametrize("name", demo.list_targets())
    def test_every_asset_loads(self, name):
        audio, params = demo.load_asset(name)
        assert audio.shape == (1, demo.SAMPLE_RATE) and audio.dtype == torch.float32
        assert params.shape == (1, demo.SAMPLE_RATE // demo.HOP, 13)
        assert params.min() >= 0 and params.max() <= 1

    @pytest.mark.parametrize("name", demo.list_targets())
    def test_every_asset_synthesizes_finite(self, name):
        _, params = demo.load_asset(name)
        torch.manual_seed(0)
        audio = demo.synthesize(params, demo.ParamToLPC())
        assert audio.shape == (1, demo.SAMPLE_RATE)
        assert torch.isfinite(audio).all()


class TestConfig:
    def test_default_init_depends_on_mode(self):
        assert demo.DemoConfig(target_mode="wav").resolved_init() == "npz"
        assert demo.DemoConfig(target_mode="self").resolved_init() == "mid"

    def test_self_mode_rejects_npz_init(self):
        with pytest.raises(ValueError):
            demo.DemoConfig(target_mode="self", init="npz").resolved_init()

    def test_main_returns_2_on_bad_config(self, out_path, capsys):
        assert demo.main(["--target-mode", "self", "--init", "npz", "--out", str(out_path)]) == 2
        assert "error" in capsys.readouterr().err

    def test_initial_params(self):
        truth = torch.rand(1, 50, 13)
        assert torch.equal(demo.initial_params("npz", truth), truth)
        assert torch.all(demo.initial_params("mid", truth) == 0.5)
        rand = demo.initial_params("random", truth)
        assert rand.shape == truth.shape and 0 <= rand.min() and rand.max() <= 1


class TestRun:
    def test_short_run_reduces_loss_and_keeps_params_in_range(self):
        result = demo.run(demo.DemoConfig(steps=30, log_every=10))
        assert len(result.history) == 31
        assert result.final_loss < result.initial_loss
        assert result.params.min() >= 0 and result.params.max() <= 1
        assert result.diverged_at is None

    def test_synthesize_is_deterministic(self):
        """Fixed-seed noise makes synthesis repeatable, so self targets are exactly recoverable."""
        torch.manual_seed(0)
        mapper = demo.ParamToLPC()
        _, truth = demo.load_asset("vt_vowel_ah")
        assert torch.equal(demo.synthesize(truth, mapper), demo.synthesize(truth, mapper))

    def test_self_mode_reduces_loss(self):
        result = demo.run(demo.DemoConfig(target_mode="self", steps=30))
        assert result.final_loss < result.initial_loss

    def test_train_mapper_flag(self):
        result = demo.run(demo.DemoConfig(steps=10, train_mapper=True, lr=1e-3))
        assert result.final_loss < result.initial_loss

    def test_stops_on_non_finite_loss(self, monkeypatch):
        real = demo.synthesize
        calls = {"n": 0}

        def flaky(params, mapper):
            calls["n"] += 1
            audio = real(params, mapper)
            return audio * float("nan") if calls["n"] > 3 else audio

        monkeypatch.setattr(demo, "synthesize", flaky)
        result = demo.run(demo.DemoConfig(steps=20))
        assert result.diverged_at == 3
        assert len(result.history) == 3


class TestMain:
    def test_writes_wavs_and_passes(self, out_path, capsys):
        assert demo.main(["--steps", "20", "--log-every", "10", "--out", str(out_path)]) == 0
        for path in (out_path, out_path.with_name("overfit_target.wav"),
                     out_path.with_name("overfit_initial.wav")):
            audio, sr = sf.read(path)
            assert sr == demo.SAMPLE_RATE and audio.shape == (demo.SAMPLE_RATE,)
        out = capsys.readouterr().out
        assert "PASS" in out and "initial loss" in out and "median gp" in out

    def test_returns_1_when_loss_does_not_decrease(self, out_path, monkeypatch):
        """lr=0 leaves params unchanged, so final == initial and the check fails."""
        assert demo.main(["--steps", "3", "--lr", "0", "--out", str(out_path)]) == 1

    def test_default_run_converges(self, out_path):
        """The full default run (500 steps) lowers the loss on average and keeps pitch near truth."""
        result = demo.run(demo.DemoConfig(out=out_path))
        first, last = demo.window_trend(result.history)
        assert last < first
        assert result.final_loss < result.initial_loss
        truth_hz = demo.median_pitch_hz(result.truth_params)
        assert abs(demo.median_pitch_hz(result.params) - truth_hz) / truth_hz < 0.05

    def test_module_entry_point(self, out_path):
        proc = subprocess.run(
            [sys.executable, "-m", "vocaltract.phase1.overfit_demo", "--steps", "5",
             "--out", str(out_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert out_path.exists()
