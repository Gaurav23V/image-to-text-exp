from __future__ import annotations

from src.llm.ollama import parse_ollama_prompt_response


def test_parse_ollama_prompt_response_handles_valid_json() -> None:
    result = parse_ollama_prompt_response(
        '{"improved_prompt":"A polished prompt.","notes":"ok"}',
        original_prompt="raw prompt",
        model_name="gemma3:4b",
    )

    assert result.improved_prompt == "A polished prompt."
    assert result.used_fallback is False


def test_parse_ollama_prompt_response_falls_back_on_bad_output() -> None:
    result = parse_ollama_prompt_response("not json", original_prompt="raw prompt", model_name="gemma3:4b")

    assert result.improved_prompt == "raw prompt"
    assert result.used_fallback is True
