from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import torch


def get_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    package_names = [
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "pydantic",
        "pandas",
        "matplotlib",
        "realesrgan",
        "basicsr",
    ]
    for package_name in package_names:
        try:
            module = __import__(package_name)
            versions[package_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package_name] = "not-installed"
    return versions


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_precision(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if device == "cuda":
        return "float16"
    return "float32"


def collect_environment_metadata(root: Path) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "git_commit": get_git_commit(root),
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "package_versions": get_package_versions(),
    }
