from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config.models import AppConfig
from src.feedback.gemini import GeminiError, MockGeminiClient, build_gemini_client
from src.io.artifacts import build_grid, ensure_directories, next_run_id, save_dataframe, save_image, save_json
from src.io.prompts import load_prompts
from src.metrics.clip_score import compute_clip_score
from src.models.adapters import ModelLoadError, build_text_to_image_adapter
from src.models.registry import get_model_spec
from src.reporting.reports import generate_feedback_reports
from src.utils.env import collect_environment_metadata, detect_device, detect_precision
from src.utils.schemas import FeedbackResult

logger = logging.getLogger(__name__)


def run_feedback_phase(config: AppConfig) -> Path:
    output_root = config.run.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_directories(
        output_root,
        ["baseline_images", "refined_images", "critiques", "comparisons", "sample_grids", "plots", "manifests"],
    )
    run_id = next_run_id("feedback")
    save_json(
        output_root / "manifests" / f"{run_id}.json",
        {
            "run_id": run_id,
            "phase": "feedback",
            "config": config.model_dump(mode="json"),
            "environment": collect_environment_metadata(Path.cwd()),
        },
    )

    prompts = load_prompts(config.run.prompts_path, config.run.prompt_categories, config.run.prompt_limit)
    device = detect_device(config.run.device)
    precision = detect_precision(device, config.run.precision)
    try:
        gemini_client = build_gemini_client(config.feedback.mode)
    except GeminiError as exc:
        if config.run.allow_mock_fallback:
            logger.warning("Falling back to mock Gemini client: %s", exc)
            gemini_client = MockGeminiClient()
        else:
            raise

    rows: list[dict] = []
    for model_alias in config.models:
        spec = get_model_spec(model_alias)
        adapter = build_text_to_image_adapter(spec=spec, device=device, precision=precision)
        try:
            adapter.load()
        except ModelLoadError as exc:
            for prompt in prompts:
                for seed in config.run.seeds:
                    rows.append(
                        FeedbackResult(
                            run_id=run_id,
                            timestamp=datetime.now(timezone.utc),
                            model_alias=model_alias,
                            prompt_id=prompt.id,
                            seed=seed,
                            original_prompt=prompt.prompt,
                            refined_prompt=prompt.prompt,
                            success=False,
                            error=str(exc),
                        ).model_dump(mode="json")
                    )
            continue

        comparison_images: list[Image.Image] = []
        comparison_captions: list[str] = []
        for prompt in prompts:
            for seed in config.run.seeds:
                try:
                    baseline = adapter.generate(
                        prompt.prompt,
                        seed,
                        config.run.width,
                        config.run.height,
                        config.run.inference_steps,
                        config.run.guidance_scale,
                        config.run.scheduler,
                    )
                    baseline_path = output_root / "baseline_images" / model_alias / f"{prompt.id}_{seed}.png"
                    save_image(baseline_path, baseline.image)
                    current_prompt = prompt.prompt
                    latest_image = baseline.image
                    critique = None
                    raw_response_path = None
                    critique_path = None
                    for iteration in range(config.feedback.iterations):
                        critique = gemini_client.critique_image(
                            prompt=current_prompt,
                            image=latest_image,
                            template=config.feedback.critique_template,
                            model_name=config.feedback.gemini_model,
                        )
                        critique_path = output_root / "critiques" / model_alias / f"{prompt.id}_{seed}_iter{iteration + 1}.json"
                        raw_response_path = output_root / "critiques" / model_alias / f"{prompt.id}_{seed}_iter{iteration + 1}.txt"
                        save_json(critique_path, critique.model_dump(mode="json"))
                        raw_response_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_response_path.write_text(critique.raw_response)
                        current_prompt = critique.corrected_prompt or current_prompt
                        refined = adapter.generate(
                            current_prompt,
                            seed,
                            config.run.width,
                            config.run.height,
                            config.run.inference_steps,
                            config.run.guidance_scale,
                            config.run.scheduler,
                        )
                        latest_image = refined.image

                    refined_path = output_root / "refined_images" / model_alias / f"{prompt.id}_{seed}.png"
                    save_image(refined_path, latest_image)
                    baseline_clip = compute_clip_score(baseline.image, prompt.prompt) if config.metrics.enable_clip_score else None
                    refined_clip = compute_clip_score(latest_image, current_prompt) if config.metrics.enable_clip_score else None
                    comparison_grid = build_grid(
                        [baseline.image, latest_image],
                        ["baseline", "refined"],
                        columns=2,
                    )
                    comparison_path = output_root / "comparisons" / f"{model_alias}_{prompt.id}_{seed}.png"
                    save_image(comparison_path, comparison_grid)
                    comparison_images.extend([baseline.image, latest_image])
                    comparison_captions.extend([f"{prompt.id}-base", f"{prompt.id}-refined"])
                    row = FeedbackResult(
                        run_id=run_id,
                        timestamp=datetime.now(timezone.utc),
                        model_alias=model_alias,
                        prompt_id=prompt.id,
                        seed=seed,
                        baseline_image_path=str(baseline_path),
                        refined_image_path=str(refined_path),
                        original_prompt=prompt.prompt,
                        refined_prompt=current_prompt,
                        critique_path=str(critique_path),
                        raw_response_path=str(raw_response_path) if raw_response_path else None,
                        baseline_clip_score=baseline_clip,
                        refined_clip_score=refined_clip,
                        clip_score_delta=(refined_clip - baseline_clip) if None not in (refined_clip, baseline_clip) else None,
                        baseline_runtime_seconds=baseline.runtime_seconds,
                        refined_runtime_seconds=refined.runtime_seconds,
                        success=True,
                    )
                except Exception as exc:
                    row = FeedbackResult(
                        run_id=run_id,
                        timestamp=datetime.now(timezone.utc),
                        model_alias=model_alias,
                        prompt_id=prompt.id,
                        seed=seed,
                        original_prompt=prompt.prompt,
                        refined_prompt=prompt.prompt,
                        success=False,
                        error=str(exc),
                    )
                rows.append(row.model_dump(mode="json"))
        if comparison_images and config.reporting.build_grids:
            grid = build_grid(comparison_images[:4], comparison_captions[:4], columns=2)
            save_image(output_root / "sample_grids" / f"{model_alias}_feedback_grid.png", grid)

    frame = pd.DataFrame(rows)
    save_dataframe(output_root / "feedback_results.csv", frame)
    generate_feedback_reports(frame, output_root)
    return output_root
