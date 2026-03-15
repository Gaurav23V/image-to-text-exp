# Troubleshooting

## `Missing HF_TOKEN for gated model access`

Set `HF_TOKEN` in `.env` or remove the gated model alias from the config.

## `GEMINI_API_KEY is required for live Gemini calls`

Set `GEMINI_API_KEY` in `.env` or switch the phase config to `feedback.mode:
mock`.

## Large model load fails on CPU

Use `tiny_sd` or `sd_turbo`, reduce prompt count, and keep `allow_mock_fallback`
enabled for smoke runs.

## Real-ESRGAN import or weight download failure

Use the PIL fallback backend:

```yaml
super_resolution:
  backend: pil
```

## CLIP score download issues

Re-run after network access is restored or disable CLIP temporarily:

```yaml
metrics:
  enable_clip_score: false
```
