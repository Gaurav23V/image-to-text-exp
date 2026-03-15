from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image


@lru_cache(maxsize=1)
def _load_clip(model_name: str = "openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def compute_clip_score(image: Image.Image, prompt: str) -> float:
    processor, model = _load_clip()
    inputs = processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt", padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        score = torch.sum(image_embeds * text_embeds, dim=-1).item()
    return float(score)


def compute_sharpness_score(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)
