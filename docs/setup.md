# Setup

## Requirements

- Python 3.11+
- Internet access for model downloads
- Optional CUDA GPU for larger models
- Optional `HF_TOKEN` for gated Hugging Face model access
- Optional `GEMINI_API_KEY` for live prompt refinement

## Install

```bash
make setup
cp .env.example .env
```

Then populate only the credentials you need:

- `GEMINI_API_KEY` for live Gemini refinement
- `HF_TOKEN` for gated models such as SD 3.5 or FLUX entries

## First run

```bash
make smoke
```

Expected output roots:

- `results/smoke/baseline/`
- `results/smoke/feedback_loop/`
- `results/smoke/super_resolution/`

## Notes

- Smoke mode uses a tiny diffusion model and mocked Gemini.
- Full configs try real model entries and log structured skips when access or hardware is missing.
