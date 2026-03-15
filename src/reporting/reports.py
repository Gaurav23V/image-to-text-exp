from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _safe_write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _save_bar_plot(series: pd.Series, title: str, ylabel: str, output_path: Path) -> None:
    if series.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    series.sort_values().plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_baseline_reports(frame: pd.DataFrame, output_root: Path) -> None:
    if frame.empty:
        _safe_write_markdown(output_root / "summary.md", ["# Baseline summary", "", "No rows were generated."])
        return

    success_frame = frame[frame["success"] == True]  # noqa: E712
    summary_lines = [
        "# Baseline summary",
        "",
        f"- Total rows: {len(frame)}",
        f"- Successful rows: {len(success_frame)}",
        f"- Failed rows: {len(frame) - len(success_frame)}",
    ]
    if not success_frame.empty:
        grouped = success_frame.groupby("model_alias").agg(
            clip_score=("clip_score", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
        summary_lines.extend(["", "## Mean metrics by model", "", grouped.round(4).to_string()])
        _save_bar_plot(grouped["clip_score"], "CLIP score by model", "CLIP score", output_root / "plots" / "clip_score_by_model.png")
        _save_bar_plot(grouped["runtime_seconds"], "Runtime by model", "seconds", output_root / "plots" / "runtime_by_model.png")
    failures = frame.groupby("model_alias")["success"].apply(lambda col: 1 - col.mean())
    _save_bar_plot(failures, "Failure rate by model", "failure rate", output_root / "plots" / "failure_rate_by_model.png")

    if not success_frame.empty:
        category_grouped = success_frame.groupby("prompt_category")["clip_score"].mean()
        _save_bar_plot(
            category_grouped,
            "CLIP score by category",
            "CLIP score",
            output_root / "plots" / "category_breakdown.png",
        )
    _safe_write_markdown(output_root / "summary.md", summary_lines)


def generate_feedback_reports(frame: pd.DataFrame, output_root: Path) -> None:
    if frame.empty:
        _safe_write_markdown(output_root / "summary.md", ["# Feedback loop summary", "", "No rows were generated."])
        return
    summary_lines = [
        "# Feedback loop summary",
        "",
        f"- Total rows: {len(frame)}",
        f"- Successful rows: {int(frame['success'].sum())}",
        f"- Mean CLIP delta: {frame['clip_score_delta'].dropna().mean():.4f}" if frame["clip_score_delta"].notna().any() else "- Mean CLIP delta: n/a",
    ]
    deltas = frame.groupby("model_alias")["clip_score_delta"].mean().dropna()
    _save_bar_plot(deltas, "Baseline vs feedback CLIP delta", "delta", output_root / "plots" / "feedback_clip_delta.png")
    _safe_write_markdown(output_root / "summary.md", summary_lines)


def generate_super_resolution_reports(frame: pd.DataFrame, output_root: Path) -> None:
    if frame.empty:
        _safe_write_markdown(output_root / "summary.md", ["# Super-resolution summary", "", "No rows were generated."])
        return
    summary_lines = [
        "# Super-resolution summary",
        "",
        f"- Total rows: {len(frame)}",
        f"- Successful rows: {int(frame['success'].sum())}",
        f"- Mean sharpness delta: {frame['sharpness_delta'].dropna().mean():.4f}" if frame["sharpness_delta"].notna().any() else "- Mean sharpness delta: n/a",
    ]
    if frame["sharpness_delta"].notna().any():
        sharpness = frame.groupby("backend")["sharpness_delta"].mean()
        _save_bar_plot(sharpness, "Sharpness delta by backend", "delta", output_root / "plots" / "sr_sharpness_delta.png")
    runtime = frame.groupby("backend")["runtime_seconds"].mean()
    _save_bar_plot(runtime, "SR runtime by backend", "seconds", output_root / "plots" / "sr_runtime_by_backend.png")
    _safe_write_markdown(output_root / "summary.md", summary_lines)
