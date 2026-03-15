# Findings: phase 3

## Researched

- Real-ESRGAN official inference script and release assets
- practical release weight selection for a reproducible default path

## Decided

- prefer `realesr-general-x4v3`
- use PIL Lanczos as an explicit deterministic fallback
- compute sharpness deltas as a practical no-reference quality signal

## Implemented

- source discovery from baseline and feedback outputs
- fallback source generation
- super-resolution adapters
- before/after comparison artifacts
- CSV, markdown, and plot reporting

## Worked

- the phase is independently runnable
- SR output tracking is separated from generation phases

## Failed or deferred

- neural SR quality depends on dependency installation and weight downloads
