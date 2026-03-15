from __future__ import annotations

from pathlib import Path

import pytest

from src.feedback.gemini import MockGeminiClient
from src.llm.ollama import BasePromptImprover
from src.services.interactive import InteractiveWorkflowService
from src.utils.schemas import PromptImprovementResult
from tests.conftest import write_config, write_prompt_file


class StaticPromptImprover(BasePromptImprover):
    def improve_prompt(self, prompt: str) -> PromptImprovementResult:
        return PromptImprovementResult(
            original_prompt=prompt,
            improved_prompt=f"{prompt} improved",
            notes="static improver",
            raw_response='{"improved_prompt":"x"}',
            model_name="test-ollama",
            used_fallback=False,
        )


def test_interactive_feedback_run_creates_artifacts(tmp_path: Path, monkeypatch) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")
    config_path = write_config(tmp_path / "config.yaml", prompts_path, tmp_path / "results")

    clip_scores = iter([0.31, 0.44])
    monkeypatch.setattr("src.services.interactive.compute_clip_score", lambda image, prompt: next(clip_scores))
    monkeypatch.setattr("src.services.interactive.build_gemini_client", lambda mode: MockGeminiClient())

    service = InteractiveWorkflowService.from_config_path(
        config_path,
        model_alias="mock_generator",
        prompt_improver=StaticPromptImprover(),
        output_root=tmp_path / "interactive",
    )
    result = service.run_feedback("raw prompt", seed=11)

    assert result.success is True
    assert result.clip_score_delta == pytest.approx(0.13)
    assert Path(result.baseline_image_path).exists()
    assert Path(result.refined_image_path).exists()
    assert Path(result.critique_path).exists()


def test_interactive_sr_run_creates_artifacts(tmp_path: Path, monkeypatch) -> None:
    prompts_path = write_prompt_file(tmp_path / "prompts.json")
    config_path = write_config(tmp_path / "config.yaml", prompts_path, tmp_path / "results")

    clip_scores = iter([0.20, 0.27])
    monkeypatch.setattr("src.services.interactive.compute_clip_score", lambda image, prompt: next(clip_scores))

    service = InteractiveWorkflowService.from_config_path(
        config_path,
        model_alias="mock_generator",
        prompt_improver=StaticPromptImprover(),
        output_root=tmp_path / "interactive",
    )
    result = service.run_super_resolution("raw prompt", seed=11)

    assert result.success is True
    assert result.backend == "pil"
    assert Path(result.baseline_image_path).exists()
    assert Path(result.upscaled_image_path).exists()
