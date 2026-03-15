from __future__ import annotations

import json
from pathlib import Path

import yaml


def write_prompt_file(path: Path) -> Path:
    payload = [
        {
            "id": "test_prompt",
            "category": "single object",
            "prompt": "A ceramic mug on a wooden table.",
            "notes": "test",
            "difficulty": "easy",
        }
    ]
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_config(path: Path, prompts_path: Path, output_root: Path, **overrides) -> Path:
    payload = {
        "run": {
            "name": "test_run",
            "output_root": str(output_root),
            "prompts_path": str(prompts_path),
            "prompt_limit": 1,
            "prompt_categories": [],
            "seeds": [11],
            "images_per_prompt": 1,
            "width": 128,
            "height": 128,
            "inference_steps": 1,
            "guidance_scale": 0.0,
            "scheduler": "DDIMScheduler",
            "device": "cpu",
            "precision": "float32",
            "smoke_mode": True,
            "allow_mock_fallback": True,
        },
        "models": ["mock_generator"],
        "metrics": {
            "enable_clip_score": False,
            "enable_sharpness": True,
            "enable_fid": False,
        },
        "reporting": {
            "build_plots": False,
            "build_grids": True,
        },
        "feedback": {
            "mode": "mock",
            "gemini_model": "gemini-2.5-flash",
            "iterations": 1,
            "critique_template": "Return JSON for {prompt}",
        },
        "super_resolution": {
            "backend": "pil",
            "fallback_backend": "pil",
            "model_name": "realesr-general-x4v3",
            "scale": 2,
            "tile": 0,
        },
    }
    for key, value in overrides.items():
        payload[key] = value
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path
