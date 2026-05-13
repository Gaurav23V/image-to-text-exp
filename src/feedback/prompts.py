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


# ── Phase 4: intelligent refinement templates ──

DEFAULT_NEGATIVE_PROMPT_TEMPLATE = """
You are an expert at writing negative prompts for Stable Diffusion image generation.
A negative prompt tells the model what to AVOID.

Given this T2I prompt: "{prompt}"

Generate a negative prompt that is SPECIFIC to this prompt. Consider:
- What objects or elements might accidentally appear that don't belong?
- What composition problems are likely (clutter, bad framing)?
- What quality issues might occur (blurry, distorted proportions)?
- What style drift might happen (wrong lighting, wrong mood)?

Rules:
- Be specific to THIS prompt, not generic
- Keep it under 150 characters
- Focus on things the model is likely to get WRONG for this specific prompt
- Do NOT include things that are obviously absent

Return strict JSON with exactly these keys:
- negative_prompt
- reasoning
""".strip()

CRITIC_TEMPLATES = {
    "composition": """
You are a COMPOSITION critic for text-to-image prompts. Focus ONLY on spatial layout.

Original prompt: "{prompt}"
Review the generated image for COMPOSITION issues:
- Object placement and spatial relationships
- Framing and camera angle
- Balance and negative space
- Depth and layering

Produce a corrected prompt that fixes composition problems while preserving the original intent.

CRITICAL: The corrected_prompt MUST be under 50 words. Be concise. Do not repeat the full original prompt — only specify what to change.
Return strict JSON with exactly these keys:
- issues
- corrected_prompt
- confidence
""".strip(),

    "detail": """
You are a DETAIL critic for text-to-image prompts. Focus ONLY on object attributes.

Original prompt: "{prompt}"
Review the generated image for DETAIL issues:
- Object attributes (color, material, shape, texture)
- Fine-grained correctness (correct number of items, specific features)
- Missing or incorrect elements mentioned in the prompt
- Textures and surface details

Produce a corrected prompt that fixes detail problems while preserving the original intent.

CRITICAL: The corrected_prompt MUST be under 50 words. Be concise. Do not repeat the full original prompt — only specify what to change.
Return strict JSON with exactly these keys:
- issues
- corrected_prompt
- confidence
""".strip(),

    "style": """
You are a STYLE critic for text-to-image prompts. Focus ONLY on aesthetic quality.

Original prompt: "{prompt}"
Review the generated image for STYLE issues:
- Lighting quality, color palette, and mood
- Artistic coherence and atmosphere
- Rendering quality (photorealistic vs artistic consistency)
- Contrast, saturation, and overall visual appeal

Produce a corrected prompt that fixes style problems while preserving the original intent.

CRITICAL: The corrected_prompt MUST be under 50 words. Be concise. Do not repeat the full original prompt — only specify what to change.
Return strict JSON with exactly these keys:
- issues
- corrected_prompt
- confidence
""".strip(),
}
