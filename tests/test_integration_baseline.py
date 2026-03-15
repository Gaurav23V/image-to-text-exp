from __future__ import annotations

from pathlib import Path

from src.config.models import load_config
from src.pipelines.baseline import execute_generation_suite
from tests.conftest import write_config, write_prompt_file


def test_baseline_pipeline_creates_artifacts(tmp_path: Path) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")
    output_root = tmp_path / "baseline"
    config_path = write_config(tmp_path / "config.yaml", prompts_path, output_root)
    config = load_config(config_path)

    frame = execute_generation_suite(config)

    assert not frame.empty
    assert (output_root / "baseline_results.csv").exists()
    assert list((output_root / "images" / "mock_generator").glob("*.png"))
    assert list((output_root / "metadata" / "mock_generator").glob("*.json"))
