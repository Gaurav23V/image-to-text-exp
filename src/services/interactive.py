from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.config.models import AppConfig, load_config
from src.feedback.gemini import GeminiError, build_gemini_client
from src.feedback.prompts import CRITIC_TEMPLATES, DEFAULT_GEMINI_CRITIQUE_TEMPLATE, DEFAULT_NEGATIVE_PROMPT_TEMPLATE
from src.io.artifacts import ensure_directories, next_run_id, save_image, save_json
from src.llm.ollama import BasePromptImprover, OllamaRefinementClient, PassthroughPromptImprover, build_prompt_improver
from src.metrics.clip_score import compute_clip_score
from src.models.adapters import BaseTextToImageAdapter, ModelLoadError, build_text_to_image_adapter
from src.models.registry import get_model_spec
from src.sr.adapters import PILUpscaler, BaseSuperResolutionAdapter, build_super_resolution_adapter
from src.utils.env import detect_device, detect_precision
from src.utils.schemas import (
    InteractiveFeedbackRun,
    InteractiveSuperResolutionRun,
    PromptImprovementResult,
    RefinementResult,
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

    def run_intelligent_refinement(self, raw_prompt: str, seed: int = 42) -> RefinementResult:
        import time

        prompt_start = time.perf_counter()
        run_id = next_run_id("intelligent_refinement")
        timestamp = datetime.now(timezone.utc)
        prompt_slug = _slugify_prompt(raw_prompt)
        output_root = self.output_root / "refinement" / run_id
        dirs = ensure_directories(output_root, ["winner", "candidates", "baseline", "critiques", "metadata"])

        # ── Step 1: Negative prompt ──
        negative_prompt = None
        use_ollama = self.config.refinement.gemini_mode == "ollama"
        if use_ollama:
            ollama_client = OllamaRefinementClient(model_name=self.config.refinement.gemini_model)
        else:
            gemini_client = build_gemini_client(self.config.refinement.gemini_mode)

        if self.config.refinement.generate_negative:
            if use_ollama:
                negative_prompt = ollama_client.generate_negative_prompt(
                    prompt=raw_prompt,
                    template=DEFAULT_NEGATIVE_PROMPT_TEMPLATE,
                )
            else:
                negative_prompt = gemini_client.generate_negative_prompt(
                    prompt=raw_prompt,
                    template=DEFAULT_NEGATIVE_PROMPT_TEMPLATE,
                    model_name=self.config.refinement.gemini_model,
                )

        # ── Step 2: Baseline ──
        baseline = self._generate_baseline(raw_prompt, seed)
        baseline_path = dirs["baseline"] / f"{prompt_slug}.png"
        save_image(baseline_path, baseline.image)
        baseline_clip = compute_clip_score(baseline.image, raw_prompt)

        # ── Step 3: Critics + best-of-N ──
        if not use_ollama:
            gemini_client = build_gemini_client(self.config.refinement.gemini_mode)
        critic_results: list[dict] = []
        all_candidates: list[dict] = []

        for critic_name in self.config.refinement.critics:
            template = CRITIC_TEMPLATES.get(critic_name, CRITIC_TEMPLATES["composition"])
            try:
                if use_ollama:
                    critique_dict = ollama_client.critique_image(
                        prompt=raw_prompt, image=baseline.image, template=template,
                    )
                    refined_prompt = critique_dict.get("corrected_prompt", raw_prompt) or raw_prompt
                    critique = critique_dict
                else:
                    critique = gemini_client.critique_image(
                        prompt=raw_prompt, image=baseline.image, template=template,
                        model_name=self.config.refinement.gemini_model,
                    )
                    refined_prompt = critique.corrected_prompt or raw_prompt
            except Exception:
                refined_prompt = raw_prompt
                critique = None

            critique_path = dirs["critiques"] / f"{prompt_slug}_{critic_name}.json"
            if critique:
                if use_ollama:
                    save_json(critique_path, critique)
                else:
                    save_json(critique_path, critique.model_dump(mode="json"))

            candidates = []
            try:
                generated = self.adapter.generate_batch(
                    prompt=refined_prompt, seed=seed,
                    width=self.config.run.width, height=self.config.run.height,
                    inference_steps=self.config.run.inference_steps,
                    guidance_scale=self.config.run.guidance_scale,
                    scheduler=self.config.run.scheduler,
                    negative_prompt=negative_prompt,
                    num_images=self.config.refinement.n_candidates,
                )
                for i, gen in enumerate(generated):
                    clip = compute_clip_score(gen.image, refined_prompt)
                    img_path = dirs["candidates"] / f"{prompt_slug}_{critic_name}_{i}.png"
                    save_image(img_path, gen.image)
                    candidates.append({"seed_offset": i, "clip_score": clip, "image_path": str(img_path), "image": gen.image})
            except Exception:
                pass

            best = max(candidates, key=lambda x: x["clip_score"] or -999) if candidates else None
            critic_results.append({
                "critic": critic_name, "refined_prompt": refined_prompt,
                "candidates": [{k: v for k, v in c.items() if k != "image"} for c in candidates],
                "best_clip_score": best["clip_score"] if best else None,
            })
            all_candidates.extend(candidates)

        # ── Step 4: Winner ──
        if all_candidates:
            winner = max(all_candidates, key=lambda x: x["clip_score"] or -999)
            winner_path = dirs["winner"] / f"{prompt_slug}.png"
            save_image(winner_path, winner["image"])
            winner_clip = winner["clip_score"]
            winner_critic = next((cr["critic"] for cr in critic_results
                                  for c in cr["candidates"] if c.get("clip_score") == winner["clip_score"]), None)
        else:
            winner_path = None; winner_clip = None; winner_critic = None

        total_runtime = time.perf_counter() - prompt_start
        result = RefinementResult(
            run_id=run_id, timestamp=timestamp, model_alias=self.model_alias, seed=seed,
            original_prompt=raw_prompt, negative_prompt=negative_prompt,
            winner_critic=winner_critic,
            winner_seed_offset=winner.get("seed_offset") if all_candidates else None,
            winner_prompt=winner_critic,
            winner_image_path=str(winner_path) if winner_path else None,
            winner_clip_score=winner_clip,
            baseline_image_path=str(baseline_path), baseline_clip_score=baseline_clip,
            clip_score_delta=(winner_clip - baseline_clip) if (winner_clip is not None and baseline_clip is not None) else None,
            total_runtime_seconds=total_runtime, critic_results=critic_results,
            success=True,
        )
        save_json(dirs["metadata"] / f"{prompt_slug}_result.json", result.model_dump(mode="json"))
        return result
