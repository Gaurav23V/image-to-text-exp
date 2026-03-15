from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

from src.config.models import load_config
from src.pipelines.baseline import execute_generation_suite, run_baseline_phase
from src.pipelines.feedback_loop import run_feedback_phase
from src.pipelines.super_resolution import run_super_resolution_phase
from src.reporting.reports import (
    generate_baseline_reports,
    generate_feedback_reports,
    generate_super_resolution_reports,
)
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


if __name__ == "__main__":
    app()
