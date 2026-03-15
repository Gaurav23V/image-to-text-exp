from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import psutil
import torch
from PIL import Image, ImageDraw

from src.models.registry import ModelSpec


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded."""


@dataclass(slots=True)
class GeneratedImage:
    image: Image.Image
    runtime_seconds: float
    process_memory_mb: float | None
    peak_gpu_memory_mb: float | None
    extra: dict[str, Any]


class BaseTextToImageAdapter(ABC):
    def __init__(self, spec: ModelSpec, device: str, precision: str) -> None:
        self.spec = spec
        self.device = device
        self.precision = precision

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        inference_steps: int,
        guidance_scale: float,
        scheduler: str,
    ) -> GeneratedImage:
        raise NotImplementedError


class MockTextToImageAdapter(BaseTextToImageAdapter):
    def load(self) -> None:
        return None

    def generate(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        inference_steps: int,
        guidance_scale: float,
        scheduler: str,
    ) -> GeneratedImage:
        digest = sha256(f"{prompt}|{seed}".encode()).hexdigest()
        bg = tuple(int(digest[offset : offset + 2], 16) for offset in (0, 2, 4))
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        draw.rectangle([(16, 16), (width - 16, height - 16)], outline="white", width=3)
        draw.text((24, 24), self.spec.alias, fill="white")
        draw.text((24, 54), prompt[:80], fill="white")
        return GeneratedImage(
            image=image,
            runtime_seconds=0.01,
            process_memory_mb=psutil.Process().memory_info().rss / (1024 * 1024),
            peak_gpu_memory_mb=None,
            extra={"mock": True, "seed": seed},
        )


class DiffusersTextToImageAdapter(BaseTextToImageAdapter):
    def __init__(self, spec: ModelSpec, device: str, precision: str) -> None:
        super().__init__(spec, device, precision)
        self.pipe: Any | None = None

    def _dtype(self) -> torch.dtype:
        if self.precision == "float16":
            return torch.float16
        if self.precision == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def load(self) -> None:
        if self.spec.requires_manual_setup:
            raise ModelLoadError(self.spec.notes or "Manual setup is required for this model.")
        if self.spec.gated and not os.getenv("HF_TOKEN"):
            raise ModelLoadError("Missing HF_TOKEN for gated model access.")

        try:
            from diffusers import DiffusionPipeline
        except Exception as exc:
            raise ModelLoadError(f"diffusers import failed: {exc}") from exc

        load_kwargs: dict[str, Any] = {"torch_dtype": self._dtype()}
        if os.getenv("HF_TOKEN"):
            load_kwargs["token"] = os.getenv("HF_TOKEN")
        try:
            pipe = DiffusionPipeline.from_pretrained(self.spec.model_id, **load_kwargs)
            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None
            if hasattr(pipe, "to"):
                pipe = pipe.to(self.device)
            self.pipe = pipe
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {self.spec.model_id}: {exc}") from exc

    def _apply_scheduler(self, scheduler_name: str) -> None:
        if not self.pipe or not getattr(self.pipe, "scheduler", None):
            return
        try:
            import diffusers

            scheduler_cls = getattr(diffusers, scheduler_name)
            self.pipe.scheduler = scheduler_cls.from_config(self.pipe.scheduler.config)
        except Exception:
            return

    def generate(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        inference_steps: int,
        guidance_scale: float,
        scheduler: str,
    ) -> GeneratedImage:
        if self.pipe is None:
            raise ModelLoadError("Model has not been loaded.")

        self._apply_scheduler(scheduler)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        process = psutil.Process()
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "generator": generator,
        }
        try:
            import time

            started = time.perf_counter()
            with torch.inference_mode():
                output = self.pipe(**call_kwargs)
            runtime_seconds = time.perf_counter() - started
        except TypeError:
            call_kwargs.pop("height", None)
            call_kwargs.pop("width", None)
            import time

            started = time.perf_counter()
            with torch.inference_mode():
                output = self.pipe(**call_kwargs)
            runtime_seconds = time.perf_counter() - started
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if self.device == "cuda" else None
        )
        image = output.images[0]
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image))
        return GeneratedImage(
            image=image,
            runtime_seconds=runtime_seconds,
            process_memory_mb=process.memory_info().rss / (1024 * 1024),
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            extra={"scheduler": scheduler},
        )


def build_text_to_image_adapter(
    spec: ModelSpec,
    device: str,
    precision: str,
) -> BaseTextToImageAdapter:
    if spec.adapter == "mock":
        return MockTextToImageAdapter(spec=spec, device=device, precision=precision)
    if spec.adapter == "diffusers":
        return DiffusersTextToImageAdapter(spec=spec, device=device, precision=precision)
    raise ModelLoadError(spec.notes or f"Unsupported adapter type: {spec.adapter}")
