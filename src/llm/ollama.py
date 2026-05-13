from __future__ import annotations

import base64
import json
import os
import re
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from src.feedback.prompts import DEFAULT_OLLAMA_PROMPT_IMPROVER_SYSTEM_PROMPT
from src.utils.schemas import PromptImprovementResult


class OllamaError(RuntimeError):
    """Raised when the local Ollama prompt improver fails."""


def _extract_json_block(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_ollama_prompt_response(text: str, original_prompt: str, model_name: str | None = None) -> PromptImprovementResult:
    try:
        parsed = _extract_json_block(text)
        improved_prompt = str(parsed.get("improved_prompt", "")).strip()
        if not improved_prompt:
            raise ValueError("Missing improved_prompt")
        return PromptImprovementResult(
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            notes=str(parsed.get("notes", "")),
            raw_response=text,
            model_name=model_name,
            used_fallback=False,
        )
    except Exception:
        return PromptImprovementResult(
            original_prompt=original_prompt,
            improved_prompt=original_prompt,
            notes="Ollama response could not be parsed. Using the original prompt.",
            raw_response=text,
            model_name=model_name,
            used_fallback=True,
        )


class BasePromptImprover(ABC):
    @abstractmethod
    def improve_prompt(self, prompt: str) -> PromptImprovementResult:
        raise NotImplementedError


class PassthroughPromptImprover(BasePromptImprover):
    def __init__(self, model_name: str = "passthrough") -> None:
        self.model_name = model_name

    def improve_prompt(self, prompt: str) -> PromptImprovementResult:
        note = "Prompt improver unavailable. Using the original prompt."
        if self.model_name == "demo-passthrough":
            note = "Ollama prompt improvement is disabled in demo mode. Using the original prompt."
        return PromptImprovementResult(
            original_prompt=prompt,
            improved_prompt=prompt,
            notes=note,
            raw_response="",
            model_name=self.model_name,
            used_fallback=True,
        )


class LocalOllamaPromptImprover(BasePromptImprover):
    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        timeout: int = 60,
        system_prompt: str = DEFAULT_OLLAMA_PROMPT_IMPROVER_SYSTEM_PROMPT,
    ) -> None:
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt

    def improve_prompt(self, prompt: str) -> PromptImprovementResult:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise OllamaError(f"Ollama request failed: {response.status_code} {response.text[:300]}")
        body = response.json()
        raw_text = str(body.get("response", ""))
        if not raw_text.strip():
            raise OllamaError("Ollama returned an empty response.")
        return parse_ollama_prompt_response(raw_text, original_prompt=prompt, model_name=self.model_name)


def build_prompt_improver() -> BasePromptImprover:
    try:
        return LocalOllamaPromptImprover()
    except Exception:
        return PassthroughPromptImprover()


# ── Phase 4: multimodal refinement via local Ollama ──

class OllamaRefinementClient:
    """Replaces Gemini for negative prompt generation + image critique using local Ollama.
    Uses keep_alive=0 to unload model from GPU after each call, freeing VRAM for SD."""

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call(self, prompt: str, system: str = "", images: list[str] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,  # unload from GPU after response
            "options": {"temperature": 0.2},
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images

        response = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        if not response.ok:
            raise OllamaError(f"Ollama request failed: {response.status_code} {response.text[:300]}")
        body = response.json()
        text = str(body.get("response", ""))
        if not text.strip():
            raise OllamaError("Ollama returned empty response.")
        return text

    def generate_negative_prompt(self, prompt: str, template: str) -> str:
        text = self._call(prompt=template.format(prompt=prompt))
        try:
            parsed = _extract_json_block(text)
            neg = str(parsed.get("negative_prompt", "")).strip()
            if neg:
                return neg
        except Exception:
            pass
        return "blurry, low quality, distorted, bad anatomy, watermark"

    def critique_image(self, prompt: str, image: Image.Image, template: str) -> dict:
        text = self._call(
            prompt=template.format(prompt=prompt),
            images=[self._image_to_base64(image)],
        )
        try:
            parsed = _extract_json_block(text)
            return {
                "issues": parsed.get("issues", []),
                "corrected_prompt": parsed.get("corrected_prompt", prompt),
                "confidence": float(parsed.get("confidence", 0.5)),
                "raw_response": text,
            }
        except Exception:
            return {
                "issues": ["Could not parse response"],
                "corrected_prompt": prompt,
                "confidence": 0.0,
                "raw_response": text,
            }
