# Findings log

## Research references checked during implementation

- Hugging Face diffusers loading and text-to-image docs
- Hugging Face diffusers evaluation guidance for CLIP/FID caveats
- Stability AI SD 3.5 announcement and Hugging Face model card
- Qwen-Image official blog and Hugging Face entry
- FLUX.1 schnell Hugging Face public entry
- Gemini generate-content and image-understanding docs
- Real-ESRGAN official repository and releases

## Current decisions

- Use diffusers as the default text-to-image framework.
- Keep all model enablement config-driven.
- Treat SD 3.5 and FLUX as gated entries with structured skip handling.
- Keep Qwen-Image in the registry but do not assume stable loader semantics.
- Default smoke mode to a tiny diffusion pipeline.
- Use CLIP score, latency, failure rate, and memory metrics now.
- Keep FID disabled until a correct reference-dataset workflow is added.
- Use Gemini through the REST `generateContent` path to avoid SDK churn.
- Attempt Real-ESRGAN first, but provide a deterministic PIL fallback.

## What worked conceptually

- Shared config and schema design across all phases
- Reusable generation path for baseline and source generation
- Mock Gemini flow for tests and smoke mode
- Standalone SR command path that can discover or generate source images

## Known risk areas

- Qwen-Image support may require upstream-specific loader logic
- heavyweight model performance is strongly hardware-dependent
- Real-ESRGAN Python dependencies may be fragile on some systems
- CLIP downloads and large model downloads increase first-run latency

## Remaining engineering work after this pass

- Add a fully correct FID pipeline with a documented reference dataset
- Add richer GPU telemetry when CUDA is available
- Add more model-specific inference knobs for SD3/FLUX/Qwen families
