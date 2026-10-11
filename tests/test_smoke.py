"""Tests for the Phase 1 gradient-flow smoke test entry point."""

import subprocess
import sys

import torch

from vocaltract.mapper import ParamToLPC
from vocaltract.params import PARAM_NAMES
from vocaltract.phase1 import smoke_test
from vocaltract.pipeline import synthesize


def test_run_smoke_test_passes():
    result = smoke_test.run_smoke_test()
    assert result.ok, result.failures
    assert result.grad_norm > 0
    assert result.shapes["audio"] == (1, smoke_test.NUM_FRAMES * smoke_test.HOP)
    assert result.shapes["lpc coeffs (frame)"] == (1, smoke_test.NUM_FRAMES, 10)


def test_deterministic():
    a, b = smoke_test.run_smoke_test(seed=3), smoke_test.run_smoke_test(seed=3)
    assert a.loss == b.loss and a.grad_norm == b.grad_norm


def test_gradient_reaches_both_paths():
    """Formant params only reach audio via the mapper; gp only via the source."""
    torch.manual_seed(0)
    params = torch.rand(1, 10, 13, requires_grad=True)
    synthesize(params, ParamToLPC(), hop=64).pow(2).mean().backward()
    for name in ("f1", "f2", "f3", "f4", "gp"):
        assert params.grad[..., PARAM_NAMES.index(name)].abs().sum() > 0, name


def test_main_returns_zero(capsys):
    assert smoke_test.main() == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "||params.grad||" in out


def test_main_returns_nonzero_on_failure(monkeypatch, capsys):
    failing = smoke_test.SmokeResult(shapes={}, loss=0.0, grad_norm=0.0, failures=["boom"])
    monkeypatch.setattr(smoke_test, "run_smoke_test", lambda: failing)
    assert smoke_test.main() == 1
    assert "FAIL: boom" in capsys.readouterr().err


def test_module_entry_point_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "vocaltract.phase1.smoke_test"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout
