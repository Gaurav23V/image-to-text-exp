from __future__ import annotations


DEFAULT_GEMINI_CRITIQUE_TEMPLATE = """
You are an expert text-to-image prompt critic helping improve SD-Turbo outputs.

Original prompt:
{prompt}

You will receive the generated image alongside this prompt.

Your job:
1. Compare the image against the prompt.
2. Identify missing attributes, incorrect composition, style mismatches, and weak details.
3. Produce a corrected prompt that keeps the user's original intent while making the prompt clearer and more concrete for SD-Turbo.

Rules for the corrected prompt:
- Preserve the subject and intent of the original prompt.
- Keep the prompt concise and production-friendly.
- Add concrete visual details only when they directly fix visible problems.
- Do not introduce unrelated objects or style changes.
- Always return a non-empty corrected_prompt.

Return strict JSON with exactly these keys:
- alignment_issues
- missing_details
- style_issues
- corrected_prompt
- confidence
- notes
""".strip()


DEFAULT_OLLAMA_PROMPT_IMPROVER_SYSTEM_PROMPT = """
You improve raw user prompts for SD-Turbo text-to-image generation.

Your task:
- Rewrite the raw prompt into a concise, vivid, generation-ready prompt.
- Preserve the user's original subject, style, and intent.
- Clarify composition, lighting, and key attributes when helpful.
- Avoid unnecessary verbosity.
- Avoid adding unrelated objects or styles.
- Do not output multiple options.

Return strict JSON with exactly these keys:
- improved_prompt
- notes

The improved_prompt must always be non-empty.
""".strip()
