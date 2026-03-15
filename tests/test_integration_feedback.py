from __future__ import annotations

from pathlib import Path

from src.config.models import load_config
from src.pipelines.feedback_loop import run_feedback_phase
from tests.conftest import write_config, write_prompt_file


def test_feedback_pipeline_creates_comparisons(tmp_path: Path) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")
    output_root = tmp_path / "feedback"
    config_path = write_config(tmp_path / "config.yaml", prompts_path, output_root)
    config = load_config(config_path)

    run_feedback_phase(config)

    assert (output_root / "feedback_results.csv").exists()
    assert list((output_root / "comparisons").glob("*.png"))
    assert list((output_root / "critiques" / "mock_generator").glob("*.json"))
