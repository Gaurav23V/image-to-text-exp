# Open Text-to-Image Benchmark and Refinement

Reproducible benchmarking, Gemini-guided prompt refinement, and super-resolution
for open-source text-to-image models.

## What this repository does

This project implements three connected but independently runnable phases:

1. Baseline benchmarking of open-source text-to-image models.
2. Gemini-based image feedback and prompt refinement.
3. Super-resolution enhancement with a Real-ESRGAN-first pipeline.

The repository is built to be reproducible, config-driven, and usable from a few
top-level commands.

## Quick start

```bash
make setup
make test
make smoke
make phase1
make phase2
make phase3
make report
```

Python CLI equivalents:

```bash
python -m src.cli phase1 --config configs/phase1.yaml
python -m src.cli phase2 --config configs/phase2.yaml
python -m src.cli phase3 --config configs/phase3.yaml
python -m src.cli smoke --config configs/smoke.yaml
python -m src.cli report --config configs/phase1.yaml
```

## Interactive demo frontend

The repository also includes a Streamlit frontend for prompt-by-prompt
interactive testing with an SD-Turbo baseline.

Features:

- raw prompt input
- optional local Ollama prompt improvement using `gemma3:4b`
- Gemini-based feedback refinement workflow
- super-resolution workflow
- side-by-side image comparisons
- CLIP score display for each output

Start the frontend:

```bash
python -m streamlit run src/frontend/app.py
```

The default sidebar config paths point to `configs/phase2.yaml` and
`configs/phase3.yaml`. For lightweight UI testing, use:

- `configs/interactive_feedback_demo.yaml`
- `configs/interactive_sr_demo.yaml`

Interactive CLI equivalents:

```bash
python -m src.cli feedback-once --prompt "A watercolor fox reading a book in a library"
python -m src.cli sr-once --prompt "A watercolor fox reading a book in a library"
python -m src.cli ui
```

## Environment

Copy `.env.example` to `.env` and update the values you need:

- `GEMINI_API_KEY` for live Gemini refinement.
- `HF_TOKEN` for gated Hugging Face models such as SD 3.5 and FLUX variants.
- `OLLAMA_HOST` and `OLLAMA_MODEL` for local prompt improvement.

The code works without those credentials in smoke mode and in mocked feedback
mode. Missing credentials are logged as structured skips instead of crashing the
whole run.

## Repository layout

- `configs/` phase configs and model settings
- `prompts/` benchmark prompt suites
- `src/` application code
- `tests/` unit and integration tests
- `docs/` practical documentation and findings logs
- `results/` generated artifacts, reports, logs, plots, and metadata

## Default model strategy

The model registry includes:

- `stabilityai/stable-diffusion-3.5-medium`
- `stabilityai/stable-diffusion-3.5-large-turbo`
- `Qwen/Qwen-Image`
- `black-forest-labs/FLUX.1-schnell`
- `Tencent/HunyuanImage-3.0` as an optional heavyweight profile
- lightweight smoke/test entries for constrained environments

The default configs are intentionally conservative:

- smoke mode uses a tiny diffusion model
- full benchmarking attempts practical public models first
- gated or heavyweight models are skipped with a clear reason when unavailable

## Outputs

Each phase writes deterministic artifacts under `results/`:

- images
- metadata JSON
- manifests
- CSV tables
- markdown summaries
- plots
- sample grids

See `docs/` for exact behavior, rerun commands, findings, and troubleshooting.
