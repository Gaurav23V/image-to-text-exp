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


def test_feedback_once_cli_outputs_paths(monkeypatch) -> None:
    class FakeService:
        def run_feedback(self, prompt: str, seed: int = 101):
            class Result:
                baseline_image_path = "baseline.png"
                refined_image_path = "refined.png"
                improved_prompt = "improved prompt"
                refined_prompt = "refined prompt"
                baseline_clip_score = 0.2
                refined_clip_score = 0.3
                clip_score_delta = 0.1

            return Result()

    monkeypatch.setattr("src.cli._load", lambda config: (Path(config), object()))
    monkeypatch.setattr("src.cli.InteractiveWorkflowService.from_config_path", lambda config: FakeService())

    result = CliRunner().invoke(app, ["feedback-once", "--prompt", "test prompt"])

    assert result.exit_code == 0
    assert "baseline.png" in result.stdout
    assert "refined.png" in result.stdout
