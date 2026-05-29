"""Tests for the skyai CLI surface"""

from __future__ import annotations

import logging
from importlib.metadata import version as _pkg_version
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from skyai.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Save/restore root logger handlers around each test so state doesnt leak"""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear
    yield

    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:
            pass
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)

def _minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimal but valid RunCOnfig YAML and return its path"""
    cfg = {
        "total_batch_size": 64,
            "model": {
                "n_layer": 2,
                "n_head": 4,
                "n_embed": 64,
                "vocab_size": 100,
                "block_size": 16,
            },
            "data": {
                "root": str(tmp_path / "shards"),
                "batch_size": 2,
            },
            "optim": {"weight_decay": 0.1},
            "schedule": {
                "max_lr": 1e-3,
                "min_lr": 1e-4,
                "warmup_steps": 10,
                "max_steps": 100,
            },
            "eval": {"interval": 50},
            "log": {"dir": str(tmp_path / "logs")},
            "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class TestVersion:
    def test_version_prints_package_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert _pkg_version("skyai") in result.output


class TestHelp:
    def test_root_help_lists_all_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for name in ("version", "train", "eval", "sample", "doctor"):
            assert name in result.output    

    def test_train_help_mentions_config_and_resume(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--resume" in result.output

    def test_eval_help_mentions_config_and_checkpoint(self) -> None:
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--checkpoint" in result.output

    def test_sample_help_mentions_checkpoint_and_prompt(self) -> None:
        result = runner.invoke(app, ["sample", "--help"])
        assert result.exit_code == 0
        assert "--checkpoint" in result.output
        assert "--prompt" in result.output


class TestErrors:
    def test_train_missing_config_errors(self) -> None:
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0

    def test_eval_missing_checkpoint_errors(self, tmp_path: Path) -> None:
        cfg = _minimal_yaml(tmp_path)
        result = runner.invoke(app, ["eval", "--config", str(cfg), "--checkpoint", str(tmp_path / "missing.pt")])
        assert result.exit_code != 0

    def test_train_nonexistent_config_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["train", "--config", str(tmp_path / "missing.yaml")])
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_train_invalid_override_errors(self, tmp_path: Path) -> None:
        cfg = _minimal_yaml(tmp_path)
        result = runner.invoke(app, ["train", "--config", str(cfg), "--override", "no-equals-here"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
