# Findings: phase 2

## Researched

- Gemini image-understanding and generate-content request format
- strict-schema prompt patterns for multimodal critique output

## Decided

- use a REST client instead of a fast-moving SDK
- support both mocked and live Gemini modes
- preserve raw Gemini output and parsed critique separately

## Implemented

- Gemini response parser with malformed-response fallback
- mock and live Gemini clients
- iterative prompt refinement loop
- baseline vs refined comparison outputs and summary tables

## Worked

- mocked mode is deterministic and suitable for tests
- malformed responses do not crash the whole phase
- live Gemini returned a usable corrected prompt once the parser accepted fenced
  JSON and string-valued issue fields

## Failed or deferred

- live quality findings depend on credential availability and rate limits
