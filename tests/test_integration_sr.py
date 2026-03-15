from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.models import load_config
from src.pipelines.baseline import execute_generation_suite
from src.pipelines.super_resolution import run_super_resolution_phase
from tests.conftest import write_config, write_prompt_file


def test_sr_pipeline_upscales_generated_image(tmp_path: Path) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")

    baseline_output = tmp_path / "baseline"
    baseline_config_path = write_config(tmp_path / "baseline.yaml", prompts_path, baseline_output)
    baseline_config = load_config(baseline_config_path)
    baseline_frame = execute_generation_suite(baseline_config)

    sr_output = tmp_path / "sr"
    sr_config_path = write_config(tmp_path / "sr.yaml", prompts_path, sr_output)
    sr_config = load_config(sr_config_path)
    run_super_resolution_phase(sr_config, source_frames=[baseline_frame, pd.DataFrame()])

    assert (sr_output / "sr_results.csv").exists()
    assert list((sr_output / "before_after").glob("*_comparison.png"))
