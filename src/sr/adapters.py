from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests
from PIL import Image


class SuperResolutionError(RuntimeError):
    """Raised when SR fails."""


class BaseSuperResolutionAdapter(ABC):
    @abstractmethod
    def upscale(self, image: Image.Image, scale: int) -> tuple[Image.Image, float]:
        raise NotImplementedError


class PILUpscaler(BaseSuperResolutionAdapter):
    def upscale(self, image: Image.Image, scale: int) -> tuple[Image.Image, float]:
        started = time.perf_counter()
        result = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
        return result, time.perf_counter() - started


class RealESRGANUpscaler(BaseSuperResolutionAdapter):
    MODEL_URLS = {
        "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }

    def __init__(self, model_name: str, weights_dir: Path, tile: int = 0, device: str = "cpu") -> None:
        self.model_name = model_name
        self.weights_dir = weights_dir
        self.tile = tile
        self.device = device

    def _ensure_weights(self) -> Path:
        if self.model_name not in self.MODEL_URLS:
            raise SuperResolutionError(f"Unsupported Real-ESRGAN model: {self.model_name}")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        path = self.weights_dir / f"{self.model_name}.pth"
        if path.exists():
            return path
        response = requests.get(self.MODEL_URLS[self.model_name], timeout=120)
        if not response.ok:
            raise SuperResolutionError(f"Failed to download weights: {response.status_code}")
        path.write_bytes(response.content)
        return path

    def upscale(self, image: Image.Image, scale: int) -> tuple[Image.Image, float]:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        except Exception as exc:
            raise SuperResolutionError(f"Real-ESRGAN dependencies are unavailable: {exc}") from exc

        weights_path = self._ensure_weights()
        if self.model_name == "realesr-general-x4v3":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
        else:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
        upsampler = RealESRGANer(
            scale=scale,
            model_path=str(weights_path),
            model=model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=0,
            half=self.device == "cuda",
        )
        started = time.perf_counter()
        output, _ = upsampler.enhance(image.convert("RGB"), outscale=scale)
        runtime = time.perf_counter() - started
        return Image.fromarray(output), runtime


def build_super_resolution_adapter(
    backend: str,
    fallback_backend: str,
    model_name: str,
    weights_dir: Path,
    tile: int,
    device: str,
) -> BaseSuperResolutionAdapter:
    if backend == "realesrgan":
        try:
            return RealESRGANUpscaler(model_name=model_name, weights_dir=weights_dir, tile=tile, device=device)
        except Exception:
            if fallback_backend == "pil":
                return PILUpscaler()
            raise
    return PILUpscaler()
