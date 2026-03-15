from __future__ import annotations

from pathlib import Path

from src.io.prompts import load_prompts
from tests.conftest import write_prompt_file


def test_load_prompts_returns_valid_records(tmp_path: Path) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")

    prompts = load_prompts(prompts_path)

    assert len(prompts) == 1
    assert prompts[0].id == "test_prompt"
    assert prompts[0].category == "single object"
