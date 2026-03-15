from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.services.interactive import InteractiveWorkflowService
from src.utils.logging import configure_logging

load_dotenv()
configure_logging()


@st.cache_resource(show_spinner=False)
def load_feedback_service(config_path: str, config_signature: str) -> InteractiveWorkflowService:
    _ = config_signature
    return InteractiveWorkflowService.from_config_path(config_path)


@st.cache_resource(show_spinner=False)
def load_sr_service(config_path: str, config_signature: str) -> InteractiveWorkflowService:
    _ = config_signature
    return InteractiveWorkflowService.from_config_path(config_path)


def _config_signature(path: str) -> str:
    return Path(path).read_text() if Path(path).exists() else path


def _render_image(path: str, label: str, clip_score: float | None, runtime: float | None = None) -> None:
    st.image(Image.open(path), caption=label, use_container_width=True)
    if clip_score is not None:
        st.metric(f"{label} CLIP", f"{clip_score:.4f}")
    if runtime is not None:
        st.caption(f"Runtime: {runtime:.3f}s")


def main() -> None:
    st.set_page_config(page_title="Interactive T2I Refinement Demo", layout="wide")
    st.title("Interactive SD-Turbo Refinement and SR Demo")
    st.caption("Raw prompt -> Ollama prompt improvement -> SD-Turbo baseline -> Gemini refinement or SR enhancement.")

    with st.sidebar:
        st.header("Run settings")
        workflow = st.radio(
            "Choose workflow",
            ("Phase 2 — Gemini feedback", "Phase 3 — Super-resolution"),
        )
        feedback_config = st.text_input("Phase 2 config", "configs/phase2.yaml")
        sr_config = st.text_input("Phase 3 config", "configs/phase3.yaml")
        seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=101, step=1)
        st.info("The UI uses the model configured in the selected YAML. Current recommended baseline: sd_turbo.")

    prompt = st.text_area(
        "Enter a prompt",
        placeholder="A cinematic portrait of a traveler standing on a rainy city street at night...",
        height=140,
    )

    if st.button("Run workflow", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt before running the workflow.")
            st.stop()

        try:
            if workflow == "Phase 2 — Gemini feedback":
                with st.spinner("Running Ollama prompt improvement, baseline generation, and Gemini refinement..."):
                    result = load_feedback_service(feedback_config, _config_signature(feedback_config)).run_feedback(
                        prompt.strip(),
                        seed=int(seed),
                    )

                left, middle, right = st.columns([1.1, 0.9, 1.1])
                with left:
                    _render_image(
                        result.baseline_image_path,
                        "Baseline image",
                        result.baseline_clip_score,
                        result.baseline_runtime_seconds,
                    )
                with middle:
                    st.subheader("Prompt flow")
                    st.markdown("**Raw prompt**")
                    st.write(result.original_prompt)
                    st.markdown("**Ollama-improved prompt**")
                    st.write(result.improved_prompt)
                    if result.prompt_improvement_notes:
                        st.caption(result.prompt_improvement_notes)
                    st.markdown("**Gemini-refined prompt**")
                    st.write(result.refined_prompt)
                    st.metric("CLIP delta", f"{(result.clip_score_delta or 0.0):.4f}")
                    if result.ollama_response_path:
                        st.caption(f"Ollama metadata: `{result.ollama_response_path}`")
                    if result.critique_path:
                        st.caption(f"Gemini critique: `{result.critique_path}`")
                with right:
                    _render_image(
                        result.refined_image_path,
                        "Gemini-refined image",
                        result.refined_clip_score,
                        result.refined_runtime_seconds,
                    )
            else:
                with st.spinner("Running Ollama prompt improvement, baseline generation, and super-resolution..."):
                    result = load_sr_service(sr_config, _config_signature(sr_config)).run_super_resolution(
                        prompt.strip(),
                        seed=int(seed),
                    )

                left, middle, right = st.columns([1.1, 0.8, 1.1])
                with left:
                    _render_image(
                        result.baseline_image_path,
                        "Baseline image",
                        result.baseline_clip_score,
                        result.baseline_runtime_seconds,
                    )
                with middle:
                    st.subheader("Prompt flow")
                    st.markdown("**Raw prompt**")
                    st.write(result.original_prompt)
                    st.markdown("**Ollama-improved prompt**")
                    st.write(result.improved_prompt)
                    if result.prompt_improvement_notes:
                        st.caption(result.prompt_improvement_notes)
                    st.metric("CLIP delta", f"{(result.clip_score_delta or 0.0):.4f}")
                    st.metric("SR backend", result.backend)
                    st.caption(f"SR runtime: {result.sr_runtime_seconds:.3f}s")
                    if result.ollama_response_path:
                        st.caption(f"Ollama metadata: `{result.ollama_response_path}`")
                with right:
                    _render_image(
                        result.upscaled_image_path,
                        "Upscaled image",
                        result.upscaled_clip_score,
                        result.sr_runtime_seconds,
                    )
        except Exception as exc:
            st.error(f"Workflow failed: {exc}")


if __name__ == "__main__":
    main()
