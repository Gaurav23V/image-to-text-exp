# Phase 1: baseline benchmarking

## Purpose

Benchmark selected open-source text-to-image models on a fixed prompt suite with
deterministic seeds, saved artifacts, and machine-readable metrics.

## Command

```bash
make phase1
```

Equivalent:

```bash
python -m src.cli phase1 --config configs/phase1.yaml
```

## What it does

- loads a typed YAML config
- loads the prompt suite from `prompts/`
- resolves model aliases from the registry
- generates images with fixed seeds
- computes CLIP score when enabled
- records latency, process memory, and GPU memory when available
- writes images, per-sample metadata, CSV tables, plots, and summaries

## Outputs

- `results/baseline/baseline_results.csv`
- `results/baseline/summary.md`
- `results/baseline/plots/`
- `results/baseline/sample_grids/`
- `results/baseline/metadata/`
- `results/baseline/manifests/`

## Current model handling

- `tiny_sd` is the default smoke-friendly model
- `sd_turbo` is the lightweight practical public model
- `sd35_medium`, `sd35_large_turbo`, and `flux_schnell` are implemented with
  gated-access handling
- `qwen_image` is registered with graceful loader failure handling because
  upstream interface expectations may shift
- `hunyuan_image_3` is tracked as an optional manual heavyweight profile

## Limitations

- FID is intentionally disabled by default because a correct reference dataset
  workflow is not yet wired in
- very large models may be skipped on CPU-only systems or without credentials
