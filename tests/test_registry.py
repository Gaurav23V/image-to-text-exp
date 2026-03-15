from __future__ import annotations

import pytest

from src.models.registry import get_model_spec


def test_registry_contains_expected_entries() -> None:
    assert get_model_spec("mock_generator").adapter == "mock"
    assert get_model_spec("sd35_medium").gated is True
    assert get_model_spec("hunyuan_image_3").requires_manual_setup is True


def test_unknown_model_alias_raises() -> None:
    with pytest.raises(KeyError):
        get_model_spec("missing-model")
