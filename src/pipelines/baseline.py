from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config.models import AppConfig
from src.io.artifacts import build_grid, ensure_directories, next_run_id, save_dataframe, save_image, save_json
from src.io.prompts import load_prompts
from src.metrics.clip_score import compute_clip_score
from src.models.adapters import ModelLoadError, build_text_to_image_adapter
from src.models.registry import get_model_spec
from src.reporting.reports import generate_baseline_reports
from src.utils.env import collect_environment_metadata, detect_device, detect_precision
from src.utils.schemas import GenerationResult, PromptRecord

logger = logging.getLogger(__name__)


def _save_model_grid(output_root: Path, model_alias: str, image_paths: list[Path]) -> None:
    if not image_paths:
        return
    images = [Image.open(path) for path in image_paths[:4]]
    captions = [path.stem for path in image_paths[:4]]
    grid = build_grid(images, captions, columns=2)
    save_image(output_root / "sample_grids" / f"{model_alias}_grid.png", grid)


def _save_manifest(config: AppConfig, output_root: Path, run_id: str, phase: str) -> None:
    payload = {
        "run_id": run_id,
        "phase": phase,
        "config": config.model_dump(mode="json"),
        "environment": collect_environment_metadata(Path.cwd()),
    }
    save_json(output_root / "manifests" / f"{run_id}.json", payload)


def execute_generation_suite(
    config: AppConfig,
    output_root: Path | None = None,
    phase_name: str = "baseline",
    aggregate_filename: str = "baseline_results.csv",
    build_reports: bool = True,
) -> pd.DataFrame:
    output_root = output_root or config.run.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_directories(output_root, ["images", "metadata", "logs", "manifests", "sample_grids", "plots"])
    run_id = next_run_id(phase_name)
    _save_manifest(config, output_root, run_id, phase_name)

    prompts = load_prompts(
        prompts_path=config.run.prompts_path,
        categories=config.run.prompt_categories,
        limit=config.run.prompt_limit,
    )
    device = detect_device(config.run.device)
    precision = detect_precision(device, config.run.precision)
    results: list[dict] = []

    for model_alias in config.models:
        spec = get_model_spec(model_alias)
        adapter = build_text_to_image_adapter(spec=spec, device=device, precision=precision)
        load_error: str | None = None
        try:
            adapter.load()
        except ModelLoadError as exc:
            load_error = str(exc)
            logger.warning("Model %s could not load: %s", model_alias, load_error)
            if config.run.allow_mock_fallback:
                fallback_spec = get_model_spec("mock_generator")
                adapter = build_text_to_image_adapter(fallback_spec, device=device, precision=precision)
                adapter.load()
        model_image_paths: list[Path] = []
        for prompt in prompts:
            for seed in config.run.seeds:
                for image_index in range(config.run.images_per_prompt):
                    timestamp = datetime.now(timezone.utc)
                    metadata_file = output_root / "metadata" / model_alias / f"{prompt.id}_{seed}_{image_index}.json"
                    if load_error and not config.run.allow_mock_fallback:
                        result = GenerationResult(
                            run_id=run_id,
                            phase=phase_name,
                            timestamp=timestamp,
                            model_alias=model_alias,
                            model_id=spec.model_id,
                            prompt_id=prompt.id,
                            prompt=prompt.prompt,
                            prompt_category=prompt.category,
                            seed=seed,
                            image_index=image_index,
                            width=config.run.width,
                            height=config.run.height,
                            scheduler=config.run.scheduler,
                            inference_steps=config.run.inference_steps,
                            guidance_scale=config.run.guidance_scale,
                            device=device,
                            precision=precision,
                            runtime_seconds=0.0,
                            process_memory_mb=None,
                            peak_gpu_memory_mb=None,
                            success=False,
                            error=load_error,
                            metadata_path=str(metadata_file),
                        )
                        save_json(metadata_file, result.model_dump(mode="json"))
                        results.append(result.model_dump(mode="json"))
                        continue
                    try:
                        generated = adapter.generate(
                            prompt=prompt.prompt,
                            seed=seed + image_index,
                            width=config.run.width,
                            height=config.run.height,
                            inference_steps=config.run.inference_steps,
                            guidance_scale=config.run.guidance_scale,
                            scheduler=config.run.scheduler,
                        )
                        image_path = output_root / "images" / model_alias / f"{prompt.id}_{seed}_{image_index}.png"
                        save_image(image_path, generated.image)
                        model_image_paths.append(image_path)
                        clip_score = compute_clip_score(generated.image, prompt.prompt) if config.metrics.enable_clip_score else None
                        result = GenerationResult(
                            run_id=run_id,
                            phase=phase_name,
                            timestamp=timestamp,
                            model_alias=model_alias,
                            model_id=spec.model_id,
                            prompt_id=prompt.id,
                            prompt=prompt.prompt,
                            prompt_category=prompt.category,
                            seed=seed,
                            image_index=image_index,
                            width=generated.image.width,
                            height=generated.image.height,
                            scheduler=config.run.scheduler,
                            inference_steps=config.run.inference_steps,
                            guidance_scale=config.run.guidance_scale,
                            device=device,
                            precision=precision,
                            runtime_seconds=generated.runtime_seconds,
                            process_memory_mb=generated.process_memory_mb,
                            peak_gpu_memory_mb=generated.peak_gpu_memory_mb,
                            image_path=str(image_path),
                            clip_score=clip_score,
                            success=True,
                            metadata_path=str(metadata_file),
                            extra={
                                **generated.extra,
                                "requested_model_alias": model_alias,
                                "requested_model_id": spec.model_id,
                                "load_error": load_error,
                            },
                        )
                    except Exception as exc:
                        result = GenerationResult(
                            run_id=run_id,
                            phase=phase_name,
                            timestamp=timestamp,
                            model_alias=model_alias,
                            model_id=spec.model_id,
                            prompt_id=prompt.id,
                            prompt=prompt.prompt,
                            prompt_category=prompt.category,
                            seed=seed,
                            image_index=image_index,
                            width=config.run.width,
                            height=config.run.height,
                            scheduler=config.run.scheduler,
                            inference_steps=config.run.inference_steps,
                            guidance_scale=config.run.guidance_scale,
                            device=device,
                            precision=precision,
                            runtime_seconds=0.0,
                            process_memory_mb=None,
                            peak_gpu_memory_mb=None,
                            success=False,
                            error=str(exc),
                            metadata_path=str(metadata_file),
                        )
                    save_json(metadata_file, result.model_dump(mode="json"))
                    results.append(result.model_dump(mode="json"))
        if config.reporting.build_grids:
            _save_model_grid(output_root, model_alias, model_image_paths)

    frame = pd.DataFrame(results)
    save_dataframe(output_root / aggregate_filename, frame)
    if build_reports:
        generate_baseline_reports(frame, output_root)
    return frame


def run_baseline_phase(config: AppConfig) -> Path:
    execute_generation_suite(config)
    return config.run.output_root


def build_source_generation_config(config: AppConfig, output_root: Path) -> AppConfig:
    new_config = copy.deepcopy(config)
    new_config.run.output_root = output_root
    return new_config
