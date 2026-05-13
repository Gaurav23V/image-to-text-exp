from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config.models import AppConfig
from src.feedback.gemini import GeminiError, MockGeminiClient, build_gemini_client
from src.feedback.prompts import CRITIC_TEMPLATES, DEFAULT_NEGATIVE_PROMPT_TEMPLATE
from src.io.artifacts import build_grid, ensure_directories, next_run_id, save_dataframe, save_image, save_json
from src.io.prompts import load_prompts
from src.llm.ollama import OllamaRefinementClient
from src.metrics.clip_score import compute_clip_score
from src.models.adapters import ModelLoadError, build_text_to_image_adapter
from src.models.registry import get_model_spec
from src.utils.env import collect_environment_metadata, detect_device, detect_precision
from src.utils.schemas import RefinementResult

logger = logging.getLogger(__name__)


def _generate_candidate_batch(
    adapter,
    prompt: str,
    seed: int,
    config: AppConfig,
    negative_prompt: str | None,
) -> list[dict]:
    """Generate N images for a single refined prompt. Returns list of {image, clip_score, seed_offset}."""
    try:
        generated = adapter.generate_batch(
            prompt=prompt,
            seed=seed,
            width=config.run.width,
            height=config.run.height,
            inference_steps=config.run.inference_steps,
            guidance_scale=config.run.guidance_scale,
            scheduler=config.run.scheduler,
            negative_prompt=negative_prompt,
            num_images=config.refinement.n_candidates,
        )
    except Exception as exc:
        logger.warning("Batch generation failed for prompt: %s", exc)
        return []

    results = []
    for i, gen in enumerate(generated):
        clip = compute_clip_score(gen.image, prompt) if config.metrics.enable_clip_score else None
        results.append({
            "image": gen.image,
            "clip_score": clip,
            "seed_offset": i,
            "runtime_seconds": gen.runtime_seconds,
        })
    return results


def run_intelligent_refinement(config: AppConfig) -> Path:
    output_root = config.run.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_directories(output_root, ["winner", "candidates", "baseline", "manifests", "plots", "critiques"])

    run_id = next_run_id("refinement")
    save_json(
        output_root / "manifests" / f"{run_id}.json",
        {
            "run_id": run_id,
            "phase": "intelligent_refinement",
            "config": config.model_dump(mode="json"),
            "environment": collect_environment_metadata(Path.cwd()),
        },
    )

    prompts = load_prompts(config.run.prompts_path, config.run.prompt_categories, config.run.prompt_limit)
    device = detect_device(config.run.device)
    precision = detect_precision(device, config.run.precision)

    # ── Build refinement client (Gemini or Ollama) ──
    if config.refinement.gemini_mode == "ollama":
        refinement_client = OllamaRefinementClient(model_name=config.refinement.gemini_model)
        use_gemini = False
    else:
        try:
            gemini_client = build_gemini_client(config.refinement.gemini_mode)
        except GeminiError as exc:
            if config.run.allow_mock_fallback:
                logger.warning("Falling back to mock Gemini client: %s", exc)
                gemini_client = MockGeminiClient()
            else:
                raise
        refinement_client = gemini_client
        use_gemini = True

    rows: list[dict] = []
    for model_alias in config.models:
        spec = get_model_spec(model_alias)
        adapter = build_text_to_image_adapter(spec=spec, device=device, precision=precision)
        try:
            adapter.load()
        except ModelLoadError as exc:
            for prompt in prompts:
                rows.append(RefinementResult(
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc),
                    model_alias=model_alias,
                    seed=0,
                    original_prompt=prompt.prompt,
                    success=False,
                    error=str(exc),
                ).model_dump(mode="json"))
            continue

        for prompt in prompts:
            import time
            prompt_start = time.perf_counter()

            try:
                # ── Step 1: Generate negative prompt ──
                negative_prompt = None
                if config.refinement.generate_negative:
                    if use_gemini:
                        negative_prompt = refinement_client.generate_negative_prompt(
                            prompt=prompt.prompt,
                            template=DEFAULT_NEGATIVE_PROMPT_TEMPLATE,
                            model_name=config.refinement.gemini_model,
                        )
                    else:
                        negative_prompt = refinement_client.generate_negative_prompt(
                            prompt=prompt.prompt,
                            template=DEFAULT_NEGATIVE_PROMPT_TEMPLATE,
                        )
                    logger.info("Negative prompt: %s", (negative_prompt or "")[:80])

                # ── Step 2: Generate baseline image ──
                baseline_seed = config.run.seeds[0] if config.run.seeds else 42
                baseline_gen = adapter.generate(
                    prompt=prompt.prompt,
                    seed=baseline_seed,
                    width=config.run.width,
                    height=config.run.height,
                    inference_steps=config.run.inference_steps,
                    guidance_scale=config.run.guidance_scale,
                    scheduler=config.run.scheduler,
                )
                baseline_path = output_root / "baseline" / f"{prompt.id}.png"
                save_image(baseline_path, baseline_gen.image)
                baseline_clip = compute_clip_score(baseline_gen.image, prompt.prompt) if config.metrics.enable_clip_score else None

                # ── Step 3: Run critics (sequential for now, can be parallelized) ──
                critic_results: list[dict] = []
                all_candidates: list[dict] = []

                for critic_name in config.refinement.critics:
                    template = CRITIC_TEMPLATES.get(critic_name, CRITIC_TEMPLATES["composition"])
                    try:
                        if use_gemini:
                            critique = refinement_client.critique_image(
                                prompt=prompt.prompt,
                                image=baseline_gen.image,
                                template=template,
                                model_name=config.refinement.gemini_model,
                            )
                            refined_prompt = critique.corrected_prompt or prompt.prompt
                        else:
                            critique_dict = refinement_client.critique_image(
                                prompt=prompt.prompt,
                                image=baseline_gen.image,
                                template=template,
                            )
                            refined_prompt = critique_dict.get("corrected_prompt", prompt.prompt) or prompt.prompt
                            critique = critique_dict  # dict, not FeedbackCritique
                    except Exception as exc:
                        logger.warning("Critic %s failed: %s", critic_name, exc)
                        refined_prompt = prompt.prompt
                        critique = None

                    # Save critique
                    critique_path = output_root / "critiques" / f"{prompt.id}_{critic_name}.json"
                    if critique:
                        if use_gemini:
                            save_json(critique_path, critique.model_dump(mode="json"))
                        else:
                            save_json(critique_path, critique)

                    # Generate N images for this critic
                    candidates = _generate_candidate_batch(
                        adapter, refined_prompt, baseline_seed, config, negative_prompt
                    )

                    # Save candidate images
                    for c in candidates:
                        img_path = output_root / "candidates" / f"{prompt.id}_{critic_name}_{c['seed_offset']}.png"
                        save_image(img_path, c["image"])
                        c["image_path"] = str(img_path)

                    best = max(candidates, key=lambda x: x["clip_score"] or -999) if candidates else None

                    critic_data = {
                        "critic": critic_name,
                        "refined_prompt": refined_prompt,
                        "candidates": [
                            {"seed_offset": c["seed_offset"], "clip_score": c["clip_score"], "image_path": c.get("image_path", "")}
                            for c in candidates
                        ],
                        "best_clip_score": best["clip_score"] if best else None,
                        "best_seed_offset": best["seed_offset"] if best else None,
                    }
                    critic_results.append(critic_data)
                    all_candidates.extend(candidates)

                # ── Step 4: Pick overall winner ──
                if all_candidates:
                    winner = max(all_candidates, key=lambda x: x["clip_score"] or -999)
                    # Find which critic produced the winner
                    winner_critic = None
                    for cr in critic_results:
                        for c in cr["candidates"]:
                            if c["seed_offset"] == winner["seed_offset"] and c["clip_score"] == winner["clip_score"]:
                                winner_critic = cr["critic"]
                                break
                        if winner_critic:
                            break

                    winner_path = output_root / "winner" / f"{prompt.id}.png"
                    save_image(winner_path, winner["image"])
                    winner_clip = winner["clip_score"]
                else:
                    winner_path = None
                    winner_clip = None
                    winner_critic = None
                    winner["seed_offset"] = None

                total_runtime = time.perf_counter() - prompt_start

                row = RefinementResult(
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc),
                    model_alias=model_alias,
                    seed=baseline_seed,
                    original_prompt=prompt.prompt,
                    negative_prompt=negative_prompt,
                    winner_critic=winner_critic,
                    winner_seed_offset=winner.get("seed_offset") if all_candidates else None,
                    winner_prompt=winner_critic,
                    winner_image_path=str(winner_path) if winner_path else None,
                    winner_clip_score=winner_clip,
                    baseline_image_path=str(baseline_path),
                    baseline_clip_score=baseline_clip,
                    clip_score_delta=(winner_clip - baseline_clip) if (winner_clip is not None and baseline_clip is not None) else None,
                    total_runtime_seconds=total_runtime,
                    critic_results=critic_results,
                    success=True,
                )
            except Exception as exc:
                row = RefinementResult(
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc),
                    model_alias=model_alias,
                    seed=0,
                    original_prompt=prompt.prompt,
                    success=False,
                    error=str(exc),
                )

            rows.append(row.model_dump(mode="json"))

    # ── Save results ──
    frame = pd.DataFrame(rows)
    save_dataframe(output_root / "refinement_results.csv", frame)

    # Generate simple summary
    success_frame = frame[frame["success"] == True]  # noqa: E712
    lines = [
        "# Intelligent Refinement Summary",
        "",
        f"- Total prompts: {len(frame)}",
        f"- Successful: {len(success_frame)}",
        f"- Failed: {len(frame) - len(success_frame)}",
    ]
    if not success_frame.empty and success_frame["clip_score_delta"].notna().any():
        mean_delta = success_frame["clip_score_delta"].dropna().mean()
        lines.append(f"- Mean CLIP delta (winner vs baseline): {mean_delta:.4f}")
    if not success_frame.empty:
        winner_counts = success_frame["winner_critic"].value_counts().to_dict()
        lines.append(f"- Winner critic distribution: {winner_counts}")

    summary_path = output_root / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")

    return output_root
