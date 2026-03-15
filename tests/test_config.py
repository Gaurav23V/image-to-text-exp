from __future__ import annotations

from pathlib import Path

import pytest

from src.config.models import load_config
from tests.conftest import write_config, write_prompt_file


def test_load_config_resolves_paths(tmp_path: Path) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")
    config_path = write_config(tmp_path / "config.yaml", prompts_path, tmp_path / "results")

    config = load_config(config_path)

    assert config.run.prompts_path == prompts_path.resolve()
    assert config.run.output_root == (tmp_path / "results").resolve()


def test_invalid_feedback_iterations_raises(tmp_path: Path) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")
    config_path = write_config(
        tmp_path / "config.yaml",
        prompts_path,
        tmp_path / "results",
        feedback={
            "mode": "mock",
            "gemini_model": "gemini-2.5-flash",
            "iterations": 0,
            "critique_template": "x",
        },
    )

    with pytest.raises(ValueError):
        load_config(config_path)
