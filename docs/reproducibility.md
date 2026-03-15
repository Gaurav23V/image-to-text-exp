# Reproducibility

## Reproducibility guarantees in this repo

- typed YAML configs
- prompt suites stored outside code
- fixed seeds
- structured output roots
- per-run manifests with environment metadata
- git commit capture when available
- CSV + JSON artifacts
- regenerate-on-demand reports

## Commands

```bash
uv run python -m src.cli phase1 --config configs/phase1.yaml
uv run python -m src.cli phase2 --config configs/phase2.yaml
uv run python -m src.cli phase3 --config configs/phase3.yaml
uv run python -m src.cli report --config configs/phase1.yaml
```

## Deterministic inputs

- prompts come from `prompts/*.json`
- model aliases come from `src/models/registry.py`
- seeds and inference settings come from `configs/*.yaml`

## Reproducibility caveats

- heavyweight model downloads can change upstream availability
- gated model access still depends on user credentials
- CLIP score depends on downloading the reference CLIP model if not cached
- Real-ESRGAN fallback mode is deterministic but not equivalent to the official
  neural SR backend
