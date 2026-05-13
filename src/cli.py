from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

from src.config.models import load_config
from src.pipelines.baseline import execute_generation_suite, run_baseline_phase
from src.pipelines.feedback_loop import run_feedback_phase
from src.pipelines.intelligent_refinement import run_intelligent_refinement
from src.pipelines.super_resolution import run_super_resolution_phase
from src.reporting.reports import (
    generate_baseline_reports,
    generate_feedback_reports,
    generate_super_resolution_reports,
)
from src.services.interactive import InteractiveWorkflowService
from src.utils.logging import configure_logging

app = typer.Typer(help="Open-source text-to-image benchmarking and refinement CLI.")


def _load(config_path: str) -> tuple[Path, object]:
    load_dotenv()
    configure_logging()
    config = load_config(config_path)
    return Path(config_path), config


@app.command()
def phase1(config: str = typer.Option("configs/phase1.yaml", exists=True, help="Path to YAML config.")) -> None:
    _, loaded = _load(config)
    output_root = run_baseline_phase(loaded)
    typer.echo(f"Phase 1 complete. Outputs written to {output_root}")


@app.command()
def phase2(config: str = typer.Option("configs/phase2.yaml", exists=True, help="Path to YAML config.")) -> None:
    _, loaded = _load(config)
    output_root = run_feedback_phase(loaded)
    typer.echo(f"Phase 2 complete. Outputs written to {output_root}")


@app.command()
def phase3(config: str = typer.Option("configs/phase3.yaml", exists=True, help="Path to YAML config.")) -> None:
    _, loaded = _load(config)
    output_root = run_super_resolution_phase(loaded)
    typer.echo(f"Phase 3 complete. Outputs written to {output_root}")


@app.command()
def smoke(config: str = typer.Option("configs/smoke.yaml", exists=True, help="Path to YAML config.")) -> None:
    _, loaded = _load(config)
    baseline_config = copy.deepcopy(loaded)
    baseline_config.run.output_root = Path("results/smoke/baseline")
    baseline_frame = execute_generation_suite(baseline_config)

    feedback_config = copy.deepcopy(loaded)
    feedback_config.run.output_root = Path("results/smoke/feedback_loop")
    feedback_output = run_feedback_phase(feedback_config)
    feedback_frame = pd.read_csv(feedback_output / "feedback_results.csv")

    sr_config = copy.deepcopy(loaded)
    sr_config.run.output_root = Path("results/smoke/super_resolution")
    run_super_resolution_phase(sr_config, source_frames=[baseline_frame, feedback_frame])

    typer.echo("Smoke run complete. Outputs written to results/smoke")


@app.command()
def report(config: str = typer.Option("configs/phase1.yaml", exists=True, help="Path to YAML config.")) -> None:
    _, loaded = _load(config)
    baseline_csv = loaded.run.output_root / "baseline_results.csv"
    if baseline_csv.exists():
        generate_baseline_reports(pd.read_csv(baseline_csv), loaded.run.output_root)
    feedback_csv = Path("results/feedback_loop/feedback_results.csv")
    if feedback_csv.exists():
        generate_feedback_reports(pd.read_csv(feedback_csv), feedback_csv.parent)
    sr_csv = Path("results/super_resolution/sr_results.csv")
    if sr_csv.exists():
        generate_super_resolution_reports(pd.read_csv(sr_csv), sr_csv.parent)
    typer.echo("Reports regenerated.")


@app.command("feedback-once")
def feedback_once(
    prompt: str = typer.Option(..., help="Raw user prompt."),
    config: str = typer.Option("configs/phase2.yaml", exists=True, help="Path to phase 2 config."),
    seed: int = typer.Option(101, help="Random seed."),
) -> None:
    _load(config)
    result = InteractiveWorkflowService.from_config_path(config).run_feedback(prompt, seed=seed)
    typer.echo(f"Baseline image: {result.baseline_image_path}")
    typer.echo(f"Refined image: {result.refined_image_path}")
    typer.echo(f"Ollama prompt: {result.improved_prompt}")
    typer.echo(f"Gemini prompt: {result.refined_prompt}")
    typer.echo(f"Baseline CLIP: {result.baseline_clip_score:.4f}")
    typer.echo(f"Refined CLIP: {result.refined_clip_score:.4f}")
    typer.echo(f"CLIP delta: {result.clip_score_delta:.4f}")


@app.command("sr-once")
def sr_once(
    prompt: str = typer.Option(..., help="Raw user prompt."),
    config: str = typer.Option("configs/phase3.yaml", exists=True, help="Path to phase 3 config."),
    seed: int = typer.Option(101, help="Random seed."),
) -> None:
    _load(config)
    result = InteractiveWorkflowService.from_config_path(config).run_super_resolution(prompt, seed=seed)
    typer.echo(f"Baseline image: {result.baseline_image_path}")
    typer.echo(f"Upscaled image: {result.upscaled_image_path}")
    typer.echo(f"Ollama prompt: {result.improved_prompt}")
    typer.echo(f"Baseline CLIP: {result.baseline_clip_score:.4f}")
    typer.echo(f"Upscaled CLIP: {result.upscaled_clip_score:.4f}")
    typer.echo(f"CLIP delta: {result.clip_score_delta:.4f}")
    typer.echo(f"SR backend: {result.backend}")


@app.command()
def ui() -> None:
    load_dotenv()
    configure_logging()
    subprocess.run(["python3", "-m", "streamlit", "run", "src/frontend/app.py"], check=True)


@app.command()
def phase4(config: str = typer.Option("configs/phase4.yaml", exists=True, help="Path to YAML config.")) -> None:
    """Run the intelligent refinement pipeline (negative prompts + multi-critic + best-of-N)."""
    _, loaded = _load(config)
    output_root = run_intelligent_refinement(loaded)
    typer.echo(f"Phase 4 complete. Outputs written to {output_root}")


@app.command("phase4-once")
def phase4_once(
    prompt: str = typer.Option(..., help="Raw user prompt."),
    config: str = typer.Option("configs/phase4.yaml", exists=True, help="Path to phase 4 config."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Run intelligent refinement on a single prompt."""
    _load(config)
    result = InteractiveWorkflowService.from_config_path(config).run_intelligent_refinement(prompt, seed=seed)
    typer.echo(f"Baseline CLIP:  {result.baseline_clip_score:.4f}" if result.baseline_clip_score else "Baseline CLIP: N/A")
    typer.echo(f"Winner CLIP:    {result.winner_clip_score:.4f}" if result.winner_clip_score else "Winner CLIP: N/A")
    typer.echo(f"CLIP delta:     {result.clip_score_delta:+.4f}" if result.clip_score_delta else "CLIP delta: N/A")
    typer.echo(f"Winner critic:  {result.winner_critic}")
    typer.echo(f"Winner image:   {result.winner_image_path}")
    typer.echo(f"Total runtime:  {result.total_runtime_seconds:.1f}s" if result.total_runtime_seconds else "Total runtime: N/A")


if __name__ == "__main__":
    app()
