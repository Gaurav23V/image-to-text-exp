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

from src.utils.schemas import FeedbackCritique


class GeminiError(RuntimeError):
    """Raised when Gemini feedback fails."""


def _extract_json_block(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_feedback_response(text: str) -> FeedbackCritique:
    try:
        parsed = _extract_json_block(text)
        critique = FeedbackCritique.model_validate(
            {
                "alignment_issues": parsed.get("alignment_issues", []),
                "missing_details": parsed.get("missing_details", []),
                "style_issues": parsed.get("style_issues", []),
                "corrected_prompt": parsed.get("corrected_prompt", ""),
                "confidence": parsed.get("confidence", 0.0),
                "notes": parsed.get("notes", ""),
                "raw_response": text,
            }
        )
        if not critique.corrected_prompt:
            critique.corrected_prompt = parsed.get("prompt", "") or "No refined prompt returned."
        return critique
    except Exception:
        return FeedbackCritique(
            corrected_prompt="No refined prompt returned.",
            notes="Gemini response could not be parsed. Falling back to original prompt.",
            raw_response=text,
        )


class BaseGeminiClient(ABC):
    @abstractmethod
    def critique_image(self, prompt: str, image: Image.Image, template: str, model_name: str) -> FeedbackCritique:
        raise NotImplementedError


class MockGeminiClient(BaseGeminiClient):
    def critique_image(self, prompt: str, image: Image.Image, template: str, model_name: str) -> FeedbackCritique:
        return FeedbackCritique(
            alignment_issues=["Mock critique used for deterministic testing."],
            missing_details=["Additional lighting detail may improve alignment."],
            style_issues=[],
            corrected_prompt=f"{prompt} Add clearer subject detail, balanced composition, and cleaner lighting.",
            confidence=0.42,
            notes="Mock Gemini response.",
            raw_response='{"corrected_prompt":"mock"}',
        )


class LiveGeminiClient(BaseGeminiClient):
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str | None = None, timeout: int = 60) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY is required for live Gemini calls.")

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def critique_image(self, prompt: str, image: Image.Image, template: str, model_name: str) -> FeedbackCritique:
        url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": template.format(prompt=prompt)},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": self._image_to_base64(image),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2},
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise GeminiError(f"Gemini request failed: {response.status_code} {response.text[:300]}")
        body = response.json()
        parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if "text" in part)
        return parse_feedback_response(text)


def build_gemini_client(mode: str) -> BaseGeminiClient:
    if mode == "mock":
        return MockGeminiClient()
    return LiveGeminiClient()
