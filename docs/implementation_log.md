# Implementation log

## Step 1: repository skeleton

- researched official docs for diffusers, SD 3.5, Qwen-Image, Gemini, and Real-ESRGAN
- decided to build a config-first Python CLI with Make targets
- implemented project metadata, configs, prompts, and package skeleton
- worked: top-level command surface is defined
- failed: no runtime validation yet
- remains: pipeline code, tests, docs, end-to-end runs

## Step 2: shared runtime contract

- researched evaluation and environment metadata needs
- decided to standardize on Pydantic schemas and run manifests
- implemented config models, prompt loader, artifact helpers, and environment capture
- worked: all phases share the same metadata shape
- failed: not yet validated against real runs
- remains: phase-specific execution and tests

## Step 3: phase implementations

- researched model availability, gating, and SR fallback practicality
- decided to use a reusable baseline generator, REST Gemini client, and Real-ESRGAN-first SR
- implemented baseline, feedback loop, and super-resolution pipelines
- worked: each phase has a runnable code path
- failed: heavyweight model paths still depend on hardware/access
- remains: tests, install validation, smoke runs, docs polish

## Step 4: runtime validation

- researched actual runtime behavior through pinned setup, automated tests, smoke
  runs, a live Gemini pass, and a standalone phase 3 run
- decided to pin the validated environment in `requirements.lock`
- implemented a parser fix for fenced Gemini JSON and string-valued issue fields
- worked: 14 automated tests passed, smoke completed, live Gemini completed, and
  standalone phase 3 completed
- failed: Real-ESRGAN Python imports were incompatible with the installed
  torchvision build, so the SR phase used the documented PIL fallback
- remains: add a known-good neural SR dependency combination or a CLI backend
  alternative such as `realesrgan-ncnn-vulkan`
