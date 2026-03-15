from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class RunConfig(BaseModel):
    name: str
    output_root: Path
    prompts_path: Path
    prompt_limit: int | None = None
    prompt_categories: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [42])
    images_per_prompt: int = 1
    width: int = 512
    height: int = 512
    inference_steps: int = 4
    guidance_scale: float = 0.0
    scheduler: str = "EulerAncestralDiscreteScheduler"
    device: str = "auto"
    precision: str = "auto"
    smoke_mode: bool = False
    allow_mock_fallback: bool = False

    @field_validator("images_per_prompt")
    @classmethod
    def validate_image_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("images_per_prompt must be >= 1")
        return value

    @field_validator("width", "height")
    @classmethod
    def validate_resolution(cls, value: int) -> int:
        if value < 64:
            raise ValueError("width and height must be >= 64")
        return value


class MetricsConfig(BaseModel):
    enable_clip_score: bool = True
    enable_sharpness: bool = False
    enable_fid: bool = False


class ReportingConfig(BaseModel):
    build_plots: bool = True
    build_grids: bool = True


class FeedbackConfig(BaseModel):
    mode: Literal["mock", "live"] = "mock"
    gemini_model: str = "gemini-2.5-flash"
    iterations: int = 1
    critique_template: str

    @field_validator("iterations")
    @classmethod
    def validate_iterations(cls, value: int) -> int:
        if value < 1:
            raise ValueError("iterations must be >= 1")
        return value


class SuperResolutionConfig(BaseModel):
    backend: Literal["realesrgan", "pil"] = "realesrgan"
    fallback_backend: Literal["realesrgan", "pil"] = "pil"
    model_name: str = "realesr-general-x4v3"
    scale: int = 4
    tile: int = 0

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, value: int) -> int:
        if value < 1:
            raise ValueError("scale must be >= 1")
        return value


class AppConfig(BaseModel):
    run: RunConfig
    models: list[str]
    metrics: MetricsConfig
    reporting: ReportingConfig
    feedback: FeedbackConfig
    super_resolution: SuperResolutionConfig

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one model alias must be configured")
        return value


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text())
    config = AppConfig.model_validate(raw)
    config.run.prompts_path = (path.parent / config.run.prompts_path).resolve() if not Path(config.run.prompts_path).is_absolute() else Path(config.run.prompts_path)
    config.run.output_root = (path.parent / config.run.output_root).resolve() if not Path(config.run.output_root).is_absolute() else Path(config.run.output_root)
    return config
