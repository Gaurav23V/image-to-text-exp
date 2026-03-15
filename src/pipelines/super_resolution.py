from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config.models import AppConfig
from src.io.artifacts import build_grid, ensure_directories, save_dataframe, save_image, save_json
from src.metrics.clip_score import compute_clip_score, compute_sharpness_score
from src.pipelines.baseline import build_source_generation_config, execute_generation_suite
from src.reporting.reports import generate_super_resolution_reports
from src.sr.adapters import PILUpscaler, SuperResolutionError, build_super_resolution_adapter
from src.utils.env import collect_environment_metadata, detect_device
from src.utils.schemas import SuperResolutionResult

logger = logging.getLogger(__name__)


def _discover_existing_sources(workspace_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    baseline_csv = workspace_root / "results" / "baseline" / "baseline_results.csv"
    if baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv)
        baseline = baseline[(baseline["success"] == True) & baseline["image_path"].notna()]  # noqa: E712
        rows.extend(
            baseline[["prompt_id", "prompt", "image_path"]]
            .rename(columns={"image_path": "source_image_path"})
            .to_dict(orient="records")
        )
    feedback_csv = workspace_root / "results" / "feedback_loop" / "feedback_results.csv"
    if feedback_csv.exists():
        feedback = pd.read_csv(feedback_csv)
        feedback = feedback[(feedback["success"] == True) & feedback["refined_image_path"].notna()]  # noqa: E712
        rows.extend(
            feedback[["prompt_id", "refined_prompt", "refined_image_path"]]
            .rename(columns={"refined_prompt": "prompt", "refined_image_path": "source_image_path"})
            .to_dict(orient="records")
        )
    return pd.DataFrame(rows)


def _build_source_frame_from_supplied_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for frame in frames:
        if frame.empty:
            continue
        if "image_path" in frame.columns:
            subset = frame[(frame["success"] == True) & frame["image_path"].notna()]  # noqa: E712
            rows.extend(
                subset[["prompt_id", "prompt", "image_path"]]
                .rename(columns={"image_path": "source_image_path"})
                .to_dict(orient="records")
            )
        elif "refined_image_path" in frame.columns:
            subset = frame[(frame["success"] == True) & frame["refined_image_path"].notna()]  # noqa: E712
            rows.extend(
                subset[["prompt_id", "refined_prompt", "refined_image_path"]]
                .rename(columns={"refined_prompt": "prompt", "refined_image_path": "source_image_path"})
                .to_dict(orient="records")
            )
    return pd.DataFrame(rows)


def run_super_resolution_phase(config: AppConfig, source_frames: list[pd.DataFrame] | None = None) -> Path:
    output_root = config.run.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_directories(output_root, ["before_after", "logs", "plots", "manifests", "weights"])
    save_json(
        output_root / "manifests" / "sr_manifest.json",
        {
            "phase": "super_resolution",
            "config": config.model_dump(mode="json"),
            "environment": collect_environment_metadata(Path.cwd()),
        },
    )

    if source_frames:
        source_frame = _build_source_frame_from_supplied_frames(source_frames)
    else:
        source_frame = _discover_existing_sources(Path.cwd())
    if source_frame.empty:
        logger.info("No existing inputs found for super-resolution; generating source images first.")
        source_output_root = output_root / "source_generation"
        source_config = build_source_generation_config(config, source_output_root)
        source_frame = execute_generation_suite(
            source_config,
            output_root=source_output_root,
            phase_name="sr_source_generation",
            aggregate_filename="baseline_results.csv",
            build_reports=False,
        )
        source_frame = _build_source_frame_from_supplied_frames([source_frame])

    device = detect_device(config.run.device)
    adapter = build_super_resolution_adapter(
        backend=config.super_resolution.backend,
        fallback_backend=config.super_resolution.fallback_backend,
        model_name=config.super_resolution.model_name,
        weights_dir=output_root / "weights",
        tile=config.super_resolution.tile,
        device=device,
    )

    rows: list[dict] = []
    before_after_images: list[Image.Image] = []
    before_after_captions: list[str] = []
    for _, source in source_frame.iterrows():
        input_path = Path(source["source_image_path"])
        prompt_text = source.get("prompt") if isinstance(source.get("prompt"), str) else None
        prompt_id = source.get("prompt_id") if isinstance(source.get("prompt_id"), str) else None
        image = Image.open(input_path)
        actual_backend = config.super_resolution.backend
        try:
            upscaled, runtime_seconds = adapter.upscale(image, config.super_resolution.scale)
        except Exception as exc:
            if config.super_resolution.fallback_backend == "pil" and not isinstance(adapter, PILUpscaler):
                logger.warning("SR backend failed for %s, falling back to PIL: %s", input_path, exc)
                actual_backend = "pil"
                adapter = PILUpscaler()
                upscaled, runtime_seconds = adapter.upscale(image, config.super_resolution.scale)
            else:
                row = SuperResolutionResult(
                    run_id="sr",
                    timestamp=datetime.now(timezone.utc),
                    backend=actual_backend,
                    input_image_path=str(input_path),
                    prompt_id=prompt_id,
                    prompt=prompt_text,
                    runtime_seconds=0.0,
                    scale=config.super_resolution.scale,
                    input_width=image.width,
                    input_height=image.height,
                    success=False,
                    error=str(exc),
                )
                rows.append(row.model_dump(mode="json"))
                continue

        output_path = output_root / "before_after" / f"{input_path.stem}_x{config.super_resolution.scale}.png"
        save_image(output_path, upscaled)
        before_after = build_grid([image, upscaled], ["before", "after"], columns=2)
        comparison_path = output_root / "before_after" / f"{input_path.stem}_comparison.png"
        save_image(comparison_path, before_after)
        before_after_images.extend([image, upscaled])
        before_after_captions.extend([f"{input_path.stem}-before", f"{input_path.stem}-after"])

        input_sharpness = compute_sharpness_score(image) if config.metrics.enable_sharpness else None
        output_sharpness = compute_sharpness_score(upscaled) if config.metrics.enable_sharpness else None
        input_clip = compute_clip_score(image, prompt_text) if (config.metrics.enable_clip_score and prompt_text) else None
        output_clip = compute_clip_score(upscaled, prompt_text) if (config.metrics.enable_clip_score and prompt_text) else None
        row = SuperResolutionResult(
            run_id="sr",
            timestamp=datetime.now(timezone.utc),
            backend=actual_backend,
            input_image_path=str(input_path),
            output_image_path=str(output_path),
            prompt_id=prompt_id,
            prompt=prompt_text,
            runtime_seconds=runtime_seconds,
            scale=config.super_resolution.scale,
            input_width=image.width,
            input_height=image.height,
            output_width=upscaled.width,
            output_height=upscaled.height,
            input_sharpness=input_sharpness,
            output_sharpness=output_sharpness,
            sharpness_delta=(output_sharpness - input_sharpness) if None not in (output_sharpness, input_sharpness) else None,
            input_clip_score=input_clip,
            output_clip_score=output_clip,
            clip_score_delta=(output_clip - input_clip) if None not in (output_clip, input_clip) else None,
            success=True,
        )
        rows.append(row.model_dump(mode="json"))

    if before_after_images:
        sample_grid = build_grid(before_after_images[:4], before_after_captions[:4], columns=2)
        save_image(output_root / "before_after" / "sample_grid.png", sample_grid)

    frame = pd.DataFrame(rows)
    save_dataframe(output_root / "sr_results.csv", frame)
    generate_super_resolution_reports(frame, output_root)
    return output_root
