from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.cli import app
from tests.conftest import write_config, write_prompt_file


def test_phase1_cli_runs_with_mock_model(tmp_path: Path) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")
    config_path = write_config(tmp_path / "config.yaml", prompts_path, tmp_path / "baseline")

    result = CliRunner().invoke(app, ["phase1", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "baseline" / "baseline_results.csv").exists()
