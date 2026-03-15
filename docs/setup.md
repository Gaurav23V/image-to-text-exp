# Setup

## Requirements

- Python 3.11+
- Internet access for model downloads
- Optional CUDA GPU for larger models
- Optional `HF_TOKEN` for gated Hugging Face model access
- Optional `GEMINI_API_KEY` for live prompt refinement

## Install

```bash
uv sync --extra dev --extra sr
cp .env.example .env
```

`uv sync --extra dev --extra sr` installs the project, development
dependencies, and super-resolution extras directly from `pyproject.toml`.

Then populate only the credentials you need:

- `GEMINI_API_KEY` for live Gemini refinement
- `HF_TOKEN` for gated models such as SD 3.5 or FLUX entries

## First run

```bash
uv run python -m src.cli smoke --config configs/smoke.yaml
```

Expected output roots:

- `results/smoke/baseline/`
- `results/smoke/feedback_loop/`
- `results/smoke/super_resolution/`

## Notes

- Smoke mode uses a tiny diffusion model and mocked Gemini.
- Full configs try real model entries and log structured skips when access or hardware is missing.
