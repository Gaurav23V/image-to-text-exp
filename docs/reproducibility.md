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
make phase1
make phase2
make phase3
make report
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
