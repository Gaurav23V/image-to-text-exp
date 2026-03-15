from __future__ import annotations

import pytest

from src.feedback.gemini import GeminiError, LiveGeminiClient, parse_feedback_response


def test_parse_feedback_response_handles_valid_json() -> None:
    critique = parse_feedback_response(
        '{"alignment_issues":["x"],"missing_details":["y"],"style_issues":[],"corrected_prompt":"refined prompt","confidence":0.9,"notes":"ok"}'
    )

    assert critique.corrected_prompt == "refined prompt"
    assert critique.alignment_issues == ["x"]


def test_parse_feedback_response_handles_malformed_response() -> None:
    critique = parse_feedback_response("not json at all")

    assert critique.corrected_prompt == "No refined prompt returned."
    assert "could not be parsed" in critique.notes


def test_parse_feedback_response_handles_code_fence_and_string_fields() -> None:
    critique = parse_feedback_response(
        """```json
{
  "alignment_issues": "N/A",
  "missing_details": "Missing background detail.",
  "style_issues": "N/A",
  "corrected_prompt": "Refined prompt here.",
  "confidence": 0.8,
  "notes": "valid"
}
```"""
    )

    assert critique.corrected_prompt == "Refined prompt here."
    assert critique.missing_details == ["Missing background detail."]
    assert critique.alignment_issues == []


def test_live_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiError):
        LiveGeminiClient()
