from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.config.models import AppConfig, load_config
from src.feedback.gemini import GeminiError, build_gemini_client
from src.feedback.prompts import DEFAULT_GEMINI_CRITIQUE_TEMPLATE
from src.io.artifacts import ensure_directories, next_run_id, save_image, save_json
from src.llm.ollama import BasePromptImprover, PassthroughPromptImprover, build_prompt_improver
from src.metrics.clip_score import compute_clip_score
from src.models.adapters import BaseTextToImageAdapter, ModelLoadError, build_text_to_image_adapter
from src.models.registry import get_model_spec
from src.sr.adapters import PILUpscaler, BaseSuperResolutionAdapter, build_super_resolution_adapter
from src.utils.env import detect_device, detect_precision
from src.utils.schemas import (
    InteractiveFeedbackRun,
    InteractiveSuperResolutionRun,
    PromptImprovementResult,
)


def _slugify_prompt(prompt: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower()).strip("_")
    return slug[:64] or "prompt"


def _fallback_prompt_improvement(prompt: str, notes: str) -> PromptImprovementResult:
    return PromptImprovementResult(
        original_prompt=prompt,
        improved_prompt=prompt,
        notes=notes,
        raw_response="",
        model_name="passthrough",
        used_fallback=True,
    )


def _resolve_feedback_template(config: AppConfig) -> str:
    template = config.feedback.critique_template.strip()
    if not template or len(template) < 60:
        return DEFAULT_GEMINI_CRITIQUE_TEMPLATE
    return template


class InteractiveWorkflowService:
    def __init__(
        self,
        config: AppConfig,
        model_alias: str | None = None,
        prompt_improver: BasePromptImprover | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.config = config
        self.model_alias = model_alias or config.models[0]
        if prompt_improver is not None:
            self.prompt_improver = prompt_improver
        elif config.run.smoke_mode:
            self.prompt_improver = PassthroughPromptImprover(model_name="demo-passthrough")
        else:
            self.prompt_improver = build_prompt_improver()
        self.output_root = output_root or (Path("results") / "interactive")
        self.device = detect_device(config.run.device)
        self.precision = detect_precision(self.device, config.run.precision)
        self.model_spec = get_model_spec(self.model_alias)
        self.adapter = self._build_generation_adapter()
        self.feedback_template = _resolve_feedback_template(config)

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        model_alias: str | None = None,
        prompt_improver: BasePromptImprover | None = None,
        output_root: Path | None = None,
    ) -> "InteractiveWorkflowService":
        config = load_config(config_path)
        return cls(config=config, model_alias=model_alias, prompt_improver=prompt_improver, output_root=output_root)

    def _build_generation_adapter(self) -> BaseTextToImageAdapter:
        adapter = build_text_to_image_adapter(self.model_spec, device=self.device, precision=self.precision)
        adapter.load()
        return adapter

    def _build_sr_adapter(self) -> BaseSuperResolutionAdapter:
        return build_super_resolution_adapter(
            backend=self.config.super_resolution.backend,
            fallback_backend=self.config.super_resolution.fallback_backend,
            model_name=self.config.super_resolution.model_name,
            weights_dir=self.output_root / "weights",
            tile=self.config.super_resolution.tile,
            device=self.device,
        )

    def _improve_prompt(self, prompt: str) -> PromptImprovementResult:
        try:
            return self.prompt_improver.improve_prompt(prompt)
        except Exception as exc:
            return _fallback_prompt_improvement(prompt, notes=f"Ollama prompt improvement failed: {exc}")

    def _generate_baseline(self, prompt: str, seed: int):
        return self.adapter.generate(
            prompt=prompt,
            seed=seed,
            width=self.config.run.width,
            height=self.config.run.height,
            inference_steps=self.config.run.inference_steps,
            guidance_scale=self.config.run.guidance_scale,
            scheduler=self.config.run.scheduler,
        )

    def run_feedback(self, raw_prompt: str, seed: int = 101) -> InteractiveFeedbackRun:
        run_id = next_run_id("interactive_feedback")
        timestamp = datetime.now(timezone.utc)
        prompt_slug = _slugify_prompt(raw_prompt)
        output_root = self.output_root / "feedback" / run_id
        dirs = ensure_directories(output_root, ["baseline", "refined", "metadata", "critiques", "prompts"])

        prompt_improvement = self._improve_prompt(raw_prompt)
        baseline = self._generate_baseline(prompt_improvement.improved_prompt, seed)
        baseline_path = dirs["baseline"] / f"{prompt_slug}.png"
        save_image(baseline_path, baseline.image)
        baseline_clip = compute_clip_score(baseline.image, prompt_improvement.improved_prompt)

        gemini_client = build_gemini_client(self.config.feedback.mode)
        critique = gemini_client.critique_image(
            prompt=prompt_improvement.improved_prompt,
            image=baseline.image,
            template=self.feedback_template,
            model_name=self.config.feedback.gemini_model,
        )
        refined_prompt = critique.corrected_prompt or prompt_improvement.improved_prompt
        refined = self._generate_baseline(refined_prompt, seed)
        refined_path = dirs["refined"] / f"{prompt_slug}.png"
        save_image(refined_path, refined.image)
        refined_clip = compute_clip_score(refined.image, refined_prompt)

        ollama_path = dirs["prompts"] / f"{prompt_slug}_ollama.json"
        critique_path = dirs["critiques"] / f"{prompt_slug}_critique.json"
        raw_critique_path = dirs["critiques"] / f"{prompt_slug}_critique.txt"
        save_json(ollama_path, prompt_improvement.model_dump(mode="json"))
        save_json(critique_path, critique.model_dump(mode="json"))
        raw_critique_path.write_text(critique.raw_response)

        result = InteractiveFeedbackRun(
            run_id=run_id,
            timestamp=timestamp,
            model_alias=self.model_alias,
            seed=seed,
            original_prompt=raw_prompt,
            improved_prompt=prompt_improvement.improved_prompt,
            prompt_improvement_notes=prompt_improvement.notes,
            prompt_improvement_used_fallback=prompt_improvement.used_fallback,
            refined_prompt=refined_prompt,
            baseline_image_path=str(baseline_path),
            refined_image_path=str(refined_path),
            ollama_response_path=str(ollama_path),
            critique_path=str(critique_path),
            raw_critique_path=str(raw_critique_path),
            baseline_clip_score=baseline_clip,
            refined_clip_score=refined_clip,
            clip_score_delta=refined_clip - baseline_clip,
            baseline_runtime_seconds=baseline.runtime_seconds,
            refined_runtime_seconds=refined.runtime_seconds,
            success=True,
        )
        save_json(dirs["metadata"] / f"{prompt_slug}_result.json", result.model_dump(mode="json"))
        return result

    def run_super_resolution(self, raw_prompt: str, seed: int = 101) -> InteractiveSuperResolutionRun:
        run_id = next_run_id("interactive_sr")
        timestamp = datetime.now(timezone.utc)
        prompt_slug = _slugify_prompt(raw_prompt)
        output_root = self.output_root / "super_resolution" / run_id
        dirs = ensure_directories(output_root, ["baseline", "upscaled", "metadata", "prompts"])

        prompt_improvement = self._improve_prompt(raw_prompt)
        baseline = self._generate_baseline(prompt_improvement.improved_prompt, seed)
        baseline_path = dirs["baseline"] / f"{prompt_slug}.png"
        save_image(baseline_path, baseline.image)
        baseline_clip = compute_clip_score(baseline.image, prompt_improvement.improved_prompt)

        adapter = self._build_sr_adapter()
        actual_backend = self.config.super_resolution.backend
        try:
            upscaled_image, sr_runtime = adapter.upscale(baseline.image, self.config.super_resolution.scale)
        except Exception as exc:
            if self.config.super_resolution.fallback_backend == "pil" and not isinstance(adapter, PILUpscaler):
                adapter = PILUpscaler()
                actual_backend = "pil"
                upscaled_image, sr_runtime = adapter.upscale(baseline.image, self.config.super_resolution.scale)
            else:
                raise exc

        upscaled_path = dirs["upscaled"] / f"{prompt_slug}_x{self.config.super_resolution.scale}.png"
        save_image(upscaled_path, upscaled_image)
        upscaled_clip = compute_clip_score(upscaled_image, prompt_improvement.improved_prompt)

        ollama_path = dirs["prompts"] / f"{prompt_slug}_ollama.json"
        save_json(ollama_path, prompt_improvement.model_dump(mode="json"))

        result = InteractiveSuperResolutionRun(
            run_id=run_id,
            timestamp=timestamp,
            model_alias=self.model_alias,
            seed=seed,
            original_prompt=raw_prompt,
            improved_prompt=prompt_improvement.improved_prompt,
            prompt_improvement_notes=prompt_improvement.notes,
            prompt_improvement_used_fallback=prompt_improvement.used_fallback,
            baseline_image_path=str(baseline_path),
            upscaled_image_path=str(upscaled_path),
            ollama_response_path=str(ollama_path),
            backend=actual_backend,
            baseline_clip_score=baseline_clip,
            upscaled_clip_score=upscaled_clip,
            clip_score_delta=upscaled_clip - baseline_clip,
            baseline_runtime_seconds=baseline.runtime_seconds,
            sr_runtime_seconds=sr_runtime,
            success=True,
        )
        save_json(dirs["metadata"] / f"{prompt_slug}_result.json", result.model_dump(mode="json"))
        return result
