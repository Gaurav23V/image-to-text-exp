from __future__ import annotations

from PIL import Image

from src.sr.adapters import PILUpscaler


def test_pil_upscaler_resizes_image() -> None:
    image = Image.new("RGB", (16, 16), "blue")

    upscaled, runtime_seconds = PILUpscaler().upscale(image, 2)

    assert upscaled.size == (32, 32)
    assert runtime_seconds >= 0.0
