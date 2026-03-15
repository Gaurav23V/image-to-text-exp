from __future__ import annotations

import json
from pathlib import Path

from src.utils.schemas import PromptRecord


def load_prompts(
    prompts_path: Path,
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[PromptRecord]:
    raw = json.loads(prompts_path.read_text())
    prompts = [PromptRecord.model_validate(item) for item in raw]
    if categories:
        prompt_set = {category.lower() for category in categories}
        prompts = [item for item in prompts if item.category.lower() in prompt_set]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts
