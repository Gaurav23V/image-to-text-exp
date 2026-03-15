from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PromptRecord(BaseModel):
    id: str
    category: str
    prompt: str
    notes: str = ""
    difficulty: str = "unknown"


class FeedbackCritique(BaseModel):
    alignment_issues: list[str] = Field(default_factory=list)
    missing_details: list[str] = Field(default_factory=list)
    style_issues: list[str] = Field(default_factory=list)
    corrected_prompt: str
    confidence: float = 0.0
    notes: str = ""
    raw_response: str = ""


class GenerationResult(BaseModel):
    run_id: str
    phase: str
    timestamp: datetime
    model_alias: str
    model_id: str
    prompt_id: str
    prompt: str
    prompt_category: str
    seed: int
    image_index: int = 0
    width: int
    height: int
    scheduler: str
    inference_steps: int
    guidance_scale: float
    device: str
    precision: str
    runtime_seconds: float
    peak_gpu_memory_mb: float | None = None
    process_memory_mb: float | None = None
    image_path: str | None = None
    clip_score: float | None = None
    sharpness_score: float | None = None
    success: bool = True
    error: str | None = None
    metadata_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class FeedbackResult(BaseModel):
    run_id: str
    timestamp: datetime
    model_alias: str
    prompt_id: str
    seed: int
    baseline_image_path: str | None = None
    refined_image_path: str | None = None
    original_prompt: str
    refined_prompt: str
    critique_path: str | None = None
    raw_response_path: str | None = None
    baseline_clip_score: float | None = None
    refined_clip_score: float | None = None
    clip_score_delta: float | None = None
    baseline_runtime_seconds: float | None = None
    refined_runtime_seconds: float | None = None
    success: bool = True
    error: str | None = None


class SuperResolutionResult(BaseModel):
    run_id: str
    timestamp: datetime
    backend: str
    input_image_path: str
    output_image_path: str | None = None
    prompt_id: str | None = None
    prompt: str | None = None
    runtime_seconds: float
    scale: int
    input_width: int
    input_height: int
    output_width: int | None = None
    output_height: int | None = None
    input_sharpness: float | None = None
    output_sharpness: float | None = None
    sharpness_delta: float | None = None
    input_clip_score: float | None = None
    output_clip_score: float | None = None
    clip_score_delta: float | None = None
    success: bool = True
    error: str | None = None
