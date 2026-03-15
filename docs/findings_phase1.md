# Findings: phase 1

## Researched

- diffusers loading patterns
- evaluation guidance for CLIP and FID
- current official/public model identifiers for SD 3.5, Qwen-Image, and FLUX

## Decided

- implement a registry with lightweight, gated, and optional heavyweight entries
- make smoke runs use `tiny_sd`
- keep FID off by default

## Implemented

- prompt suite loader
- baseline generation runner
- CLIP score metric
- artifact and manifest writing
- CSV export, summary markdown, and plots

## Worked

- registry abstraction covers both real and mock adapters
- reporting stays phase-local under `results/baseline/`

## Failed or deferred

- full FID workflow deferred
- Hunyuan kept as manual setup only
