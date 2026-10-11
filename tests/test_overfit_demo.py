"""Tests for the Phase 1 single-clip overfit demo."""

import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from vocaltract.formants import formant_error_hz, measure_formants
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
        audio = demo.synthesize(params, demo.build_mapper("formant"))
        assert audio.shape == (1, demo.SAMPLE_RATE)
        assert torch.isfinite(audio).all()


class TestConfig:
    def test_defaults_depend_on_mode(self):
        wav, self_mode = demo.DemoConfig(target_mode="wav"), demo.DemoConfig(target_mode="self")
        assert wav.resolved_init() == "npz" and self_mode.resolved_init() == "perturbed"
        assert wav.resolved_train_mapper() and not self_mode.resolved_train_mapper()
        assert wav.resolved_output_gain() and not self_mode.resolved_output_gain()
        assert wav.resolved_noise_floor() and not self_mode.resolved_noise_floor()
        assert wav.mapper == "formant" and wav.source == "engine"
        assert demo.DemoConfig(target_mode="self", train_mapper=True).resolved_train_mapper()

    def test_parse_freeze(self):
        assert demo.parse_freeze("aa=0, fa") == {"aa": 0.0, "fa": None}
        assert demo.parse_freeze("") == {}
        with pytest.raises(ValueError):
            demo.parse_freeze("xx=1")

    @pytest.mark.parametrize("source,rate", [("engine", 20000), ("vocaltract", 16000),
                                             ("sawtooth", 16000)])
    def test_build_mapper_matches_render_rate(self, source, rate):
        assert demo.build_mapper("formant", source).sample_rate == rate
        params = torch.rand(1, 3, 13)
        assert demo.synthesize(params, demo.build_mapper("formant", source), source).shape == (1, 960)

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
        near = demo.initial_params("perturbed", truth, perturb=0.05)
        assert (near - truth).abs().max() <= 0.05 + 1e-7
        assert torch.equal(near, demo.initial_params("perturbed", truth, perturb=0.05))


class TestRun:
    def test_short_run_reduces_loss_and_keeps_params_in_range(self):
        # speech_male's npz params are approximate, so short runs have room to improve.
        # (On the VocalTract targets the npz start is nearly optimal and the first
        # Adam steps go uphill; full runs recover, short ones may not.)
        result = demo.run(demo.DemoConfig(target="speech_male", steps=30, log_every=10))
        assert len(result.history) == 31
        assert result.final_loss < result.initial_loss
        assert result.final_loss == min(result.history)  # best step is reported
        assert result.params.min() >= 0 and result.params.max() <= 1
        assert result.diverged_at is None

    def test_freeze_holds_columns(self):
        result = demo.run(demo.DemoConfig(steps=15, freeze={"aa": 0.0, "ga": None}))
        aa, ga = demo.PARAM_NAMES.index("aa"), demo.PARAM_NAMES.index("ga")
        assert torch.all(result.params[..., aa] == 0.0)
        assert torch.equal(result.params[..., ga], result.truth_params[..., ga])

    def test_pitch_lr_scale_zero_freezes_pitch(self):
        result = demo.run(demo.DemoConfig(steps=15, pitch_lr_scale=0.0))
        gp = demo.GP_INDEX
        assert torch.equal(result.params[..., gp], result.truth_params[..., gp])

    def test_frame_params_round_trip(self):
        start = torch.rand(1, 50, 13)
        assert torch.equal(demo.FrameParams(start).assemble(), start)

    def test_synthesize_is_deterministic(self):
        """Fixed-seed noise makes synthesis repeatable, so self targets are exactly recoverable."""
        torch.manual_seed(0)
        mapper = demo.build_mapper("formant")
        _, truth = demo.load_asset("vt_vowel_ah")
        assert torch.equal(demo.synthesize(truth, mapper), demo.synthesize(truth, mapper))

    def test_self_mode_reduces_loss(self):
        result = demo.run(demo.DemoConfig(target_mode="self", steps=30))
        assert result.final_loss < result.initial_loss

    def test_self_mode_from_truth_is_zero_loss(self):
        """With no perturbation the self-rendered target is matched exactly."""
        result = demo.run(demo.DemoConfig(target_mode="self", init="perturbed", perturb=0.0, steps=1))
        assert result.initial_loss == pytest.approx(0.0, abs=1e-6)

    def test_train_mapper_flag(self):
        result = demo.run(demo.DemoConfig(target="speech_male", steps=10, train_mapper=True))
        assert result.final_loss < result.initial_loss

    def test_stops_on_non_finite_loss(self, monkeypatch):
        real = demo.synthesize
        calls = {"n": 0}

        def flaky(params, mapper, source="engine"):
            calls["n"] += 1
            audio = real(params, mapper, source)
            return audio * float("nan") if calls["n"] > 3 else audio

        monkeypatch.setattr(demo, "synthesize", flaky)
        result = demo.run(demo.DemoConfig(steps=20))
        assert result.diverged_at == 3
        assert len(result.history) == 3


class TestMain:
    def test_writes_wavs_and_passes(self, out_path, capsys):
        assert demo.main(["--target", "speech_male", "--steps", "20", "--log-every", "10",
                          "--out", str(out_path)]) == 0
        for path in (out_path, out_path.with_name("overfit_target.wav"),
                     out_path.with_name("overfit_initial.wav")):
            audio, sr = sf.read(path)
            assert sr == demo.SAMPLE_RATE and audio.shape == (demo.SAMPLE_RATE,)
        out = capsys.readouterr().out
        assert "PASS" in out and "loss: initial" in out and "median gp" in out
        assert "log-spectral distance" in out and "formant error measured from audio" in out

    def test_self_mode_reports_param_error(self, out_path, capsys):
        assert demo.main(["--target-mode", "self", "--steps", "10", "--out", str(out_path)]) in (0, 1)
        out = capsys.readouterr().out
        assert "param error vs truth" in out and "pitch error vs truth" in out

    def test_returns_1_when_loss_does_not_decrease(self, out_path, monkeypatch):
        """lr=0 leaves everything unchanged, so the best loss equals the initial loss."""
        assert demo.main(["--steps", "3", "--lr", "0", "--out", str(out_path)]) == 1

    @pytest.mark.slow
    def test_glide_tracks_formants(self, out_path):
        """
        Regression: with the MLP mapper, vt_glide_ah_ee held one compromise vowel
        (F2 stuck near 1460 Hz) while the target glided /a/ -> /i/. A full default
        run must follow the glide: F1 falls, F2 rises ~1100 -> ~2270 Hz, each
        within a few tens of Hz of the target, and the parameters stay near the
        engine's ground truth.
        """
        result = demo.run(demo.DemoConfig(target="vt_glide_ah_ee", out=out_path))
        assert result.final_loss < result.initial_loss
        first, last = demo.window_trend(result.history)
        assert last < first

        ours, target = measure_formants(result.final_audio), measure_formants(result.target)
        start, end = np.nanmedian(ours[3:8], axis=0), np.nanmedian(ours[-6:-1], axis=0)
        assert start[1] < 1200 and end[1] > 2150  # F2 glides up
        assert end[0] < start[0] - 250  # F1 falls
        assert np.all(formant_error_hz(result.final_audio, result.target) < 60)
        assert np.nanmax(np.abs(ours - target)[3:-1, :2]) < 100  # F1/F2 at every frame

        error_hz = demo.formant_param_error_hz(result.params, result.truth_params)
        assert max(error_hz[:3]) < 40  # f1-f3 stay near the engine's ground truth
        truth_hz = demo.median_pitch_hz(result.truth_params)
        assert abs(demo.median_pitch_hz(result.params) - truth_hz) / truth_hz < 0.01
        lsd_before = demo.log_spectral_distance(result.initial_audio, result.target)
        lsd_after = demo.log_spectral_distance(result.final_audio, result.target)
        assert lsd_after < 0.75 * lsd_before

    @pytest.mark.slow
    def test_self_mode_recovers_parameters(self, out_path):
        """
        Self-consistent target, start perturbed by 0.05 (~140 Hz per formant):
        optimization must bring the parameters back. (The MLP mapper could not:
        its parameter error grew while the loss fell.)
        """
        result = demo.run(demo.DemoConfig(target="vt_glide_ah_ee", target_mode="self",
                                          steps=300, out=out_path))
        skip = set(demo.PARAM_NAMES) - set(result.active)
        before = demo.param_error(result.initial_params, result.truth_params, skip)
        after = demo.param_error(result.params, result.truth_params, skip)
        assert after < 0.25 * before
        assert max(demo.formant_param_error_hz(result.params, result.truth_params)) < 40

    def test_module_entry_point(self, out_path):
        proc = subprocess.run(
            [sys.executable, "-m", "vocaltract.phase1.overfit_demo", "--steps", "5",
             "--target", "speech_male", "--out", str(out_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert out_path.exists()


class TestRecording:
    def test_gain_starts_at_rms_ratio(self):
        target = torch.randn(1, 1000) * 0.5
        recording = demo.Recording(target, noise_floor=False)
        audio = torch.randn(1, 1000) * 0.05
        out = recording(audio)
        assert out.pow(2).mean().sqrt().item() == pytest.approx(
            target.pow(2).mean().sqrt().item(), rel=1e-5)
        assert recording.parameters() == [recording.log_gain]

    def test_calibrates_once(self):
        recording = demo.Recording(torch.ones(1, 10), noise_floor=False)
        recording(torch.full((1, 10), 0.5))
        gain = recording.gain
        recording(torch.full((1, 10), 0.1))
        assert recording.gain == gain

    def test_noise_floor_added_and_learnable(self):
        recording = demo.Recording(torch.ones(1, 1000), noise_floor=True)
        audio = torch.ones(1, 1000)
        out = recording(audio)
        residual = out - audio * recording.gain
        assert residual.std().item() == pytest.approx(demo.Recording.INITIAL_FLOOR, rel=0.1)
        out.sum().backward()
        assert recording.log_floor.grad is not None

    def test_run_without_gain_or_floor(self):
        result = demo.run(demo.DemoConfig(target="speech_male", steps=3, output_gain=False,
                                          noise_floor=False))
        assert len(result.history) == 4
