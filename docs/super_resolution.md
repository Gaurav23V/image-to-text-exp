# Phase 3: super-resolution

## Purpose

Upscale baseline or feedback-loop outputs with a Real-ESRGAN-first pipeline and
track runtime and image-quality deltas.

## Command

```bash
uv run python -m src.cli phase3 --config configs/phase3.yaml
```

## How it chooses inputs

1. existing `results/baseline/baseline_results.csv`
2. existing `results/feedback_loop/feedback_results.csv`
3. fallback source generation if neither exists

This makes the phase runnable by itself.

## Backends

- primary: `realesrgan`
- fallback: `pil`

The default config attempts the official Real-ESRGAN path first and falls back
to a deterministic PIL Lanczos upscaler when dependencies or weights are not
available.

## Outputs

- `results/super_resolution/sr_results.csv`
- `results/super_resolution/before_after/`
- `results/super_resolution/summary.md`
- `results/super_resolution/plots/`
- `results/super_resolution/manifests/`
