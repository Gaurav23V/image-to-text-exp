from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelSpec:
    alias: str
    model_id: str
    source: str
    description: str
    family: str
    adapter: str = "diffusers"
    gated: bool = False
    optional: bool = False
    requires_manual_setup: bool = False
    recommended_steps: int | None = None
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "tiny_sd": ModelSpec(
        alias="tiny_sd",
        model_id="hf-internal-testing/tiny-stable-diffusion-pipe",
        source="huggingface",
        description="Tiny Stable Diffusion pipeline for smoke tests and constrained environments.",
        family="stable-diffusion",
        adapter="diffusers",
        recommended_steps=2,
        notes="Primary smoke/integration model. Uses the diffusers tiny pipeline commonly used for tests.",
    ),
    "sd_turbo": ModelSpec(
        alias="sd_turbo",
        model_id="stabilityai/sd-turbo",
        source="huggingface",
        description="Fast public Stability AI model for practical lightweight runs.",
        family="stable-diffusion",
        adapter="diffusers",
        recommended_steps=2,
    ),
    "sd35_medium": ModelSpec(
        alias="sd35_medium",
        model_id="stabilityai/stable-diffusion-3.5-medium",
        source="huggingface",
        description="Stable Diffusion 3.5 Medium official release.",
        family="sd3.5",
        adapter="diffusers",
        gated=True,
        recommended_steps=28,
    ),
    "sd35_large_turbo": ModelSpec(
        alias="sd35_large_turbo",
        model_id="stabilityai/stable-diffusion-3.5-large-turbo",
        source="huggingface",
        description="Stable Diffusion 3.5 Large Turbo official release.",
        family="sd3.5",
        adapter="diffusers",
        gated=True,
        optional=True,
        recommended_steps=4,
    ),
    "flux_schnell": ModelSpec(
        alias="flux_schnell",
        model_id="black-forest-labs/FLUX.1-schnell",
        source="huggingface",
        description="Black Forest Labs FLUX.1 schnell public model entry.",
        family="flux",
        adapter="diffusers",
        gated=True,
        recommended_steps=4,
    ),
    "qwen_image": ModelSpec(
        alias="qwen_image",
        model_id="Qwen/Qwen-Image",
        source="huggingface",
        description="Qwen official open image generation release.",
        family="qwen-image",
        adapter="diffusers",
        optional=True,
        recommended_steps=20,
        notes="Interface support may change across upstream releases; loader failures are preserved as structured skips.",
    ),
    "hunyuan_image_3": ModelSpec(
        alias="hunyuan_image_3",
        model_id="Tencent/HunyuanImage-3.0",
        source="github",
        description="Optional heavyweight Tencent HunyuanImage-3.0 profile.",
        family="hunyuan",
        adapter="external",
        optional=True,
        requires_manual_setup=True,
        notes="Official repository indicates heavyweight hardware and manual download/setup steps.",
    ),
    "mock_generator": ModelSpec(
        alias="mock_generator",
        model_id="mock://text-to-image",
        source="internal",
        description="Deterministic mock image generator for tests.",
        family="mock",
        adapter="mock",
        recommended_steps=1,
    ),
}


def get_model_spec(alias: str) -> ModelSpec:
    if alias not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model alias: {alias}")
    return MODEL_REGISTRY[alias]
