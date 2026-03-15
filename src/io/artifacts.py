from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageOps, ImageDraw


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_directories(root: Path, names: Iterable[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in names:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def save_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_grid(images: list[Image.Image], captions: list[str], columns: int = 2) -> Image.Image:
    if not images:
        raise ValueError("No images were provided for grid creation")
    thumb_width, thumb_height = images[0].size
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x = col * thumb_width
        y = row * (thumb_height + 28)
        canvas.paste(ImageOps.contain(image.convert("RGB"), (thumb_width, thumb_height)), (x, y))
        draw.text((x + 8, y + thumb_height + 4), captions[index][:48], fill="black")
    return canvas


def next_run_id(prefix: str) -> str:
    return f"{prefix}_{utc_timestamp()}"
