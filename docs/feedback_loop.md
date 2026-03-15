# Phase 2: Gemini feedback loop

## Purpose

Generate a baseline image, critique it with Gemini using the original prompt and
the image, create a refined prompt, regenerate, and compare before versus after.

## Command

```bash
make phase2
```

Equivalent:

```bash
python -m src.cli phase2 --config configs/phase2.yaml
```

Minimal live API check:

```bash
python -m src.cli phase2 --config configs/phase2_live_smoke.yaml
```

## Modes

- `mock` for tests and smoke mode
- `live` for real Gemini calls when `GEMINI_API_KEY` is set

## What is saved

- baseline image
- refined image
- raw Gemini response
- parsed critique JSON
- comparison grids
- CSV table with baseline/refined metrics and deltas
- markdown summary and plots

## Failure handling

- malformed Gemini output is parsed defensively
- missing API key raises a clear error unless mock fallback is enabled
- model load failures become structured row-level failures instead of stopping
  the whole run
