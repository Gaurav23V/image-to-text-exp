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


@st.cache_resource(show_spinner=False)
def load_phase4_service(config_path: str, config_signature: str) -> InteractiveWorkflowService:
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
    st.title("Interactive SD-Turbo / SD 1.5 Refinement Demo")
    st.caption("Prompt → Negative prompt → Multi-critic refinement → Best-of-N → Winner selection.")

    with st.sidebar:
        st.header("Run settings")
        workflow = st.radio(
            "Choose workflow",
            ("Phase 2 — Gemini feedback", "Phase 3 — Super-resolution", "Phase 4 — Intelligent Refinement"),
        )
        feedback_config = st.text_input("Phase 2 config", "configs/phase2.yaml")
        sr_config = st.text_input("Phase 3 config", "configs/phase3.yaml")
        phase4_config = st.text_input("Phase 4 config", "configs/phase4.yaml")
        seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

    prompt = st.text_area(
        "Enter a prompt",
        placeholder="A bright red modern chair on a seamless white background, soft shadows...",
        height=140,
    )

    if st.button("Run workflow", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt before running the workflow.")
            st.stop()

        try:
            if workflow == "Phase 2 — Gemini feedback":
                with st.spinner("Running..."):
                    result = load_feedback_service(feedback_config, _config_signature(feedback_config)).run_feedback(
                        prompt.strip(), seed=int(seed),
                    )
                left, middle, right = st.columns([1.1, 0.9, 1.1])
                with left:
                    _render_image(result.baseline_image_path, "Baseline image", result.baseline_clip_score, result.baseline_runtime_seconds)
                with middle:
                    st.subheader("Prompt flow")
                    st.write(f"**Raw:** {result.original_prompt}")
                    st.write(f"**Ollama:** {result.improved_prompt}")
                    st.write(f"**Gemini:** {result.refined_prompt}")
                    st.metric("CLIP delta", f"{(result.clip_score_delta or 0.0):.4f}")
                with right:
                    _render_image(result.refined_image_path, "Gemini-refined image", result.refined_clip_score, result.refined_runtime_seconds)

            elif workflow == "Phase 3 — Super-resolution":
                with st.spinner("Running..."):
                    result = load_sr_service(sr_config, _config_signature(sr_config)).run_super_resolution(
                        prompt.strip(), seed=int(seed),
                    )
                left, middle, right = st.columns([1.1, 0.8, 1.1])
                with left:
                    _render_image(result.baseline_image_path, "Baseline image", result.baseline_clip_score, result.baseline_runtime_seconds)
                with middle:
                    st.subheader("Prompt flow")
                    st.write(f"**Raw:** {result.original_prompt}")
                    st.write(f"**Ollama:** {result.improved_prompt}")
                    st.metric("CLIP delta", f"{(result.clip_score_delta or 0.0):.4f}")
                    st.metric("SR backend", result.backend)
                with right:
                    _render_image(result.upscaled_image_path, "Upscaled image", result.upscaled_clip_score, result.sr_runtime_seconds)

            else:  # Phase 4
                service = load_phase4_service(phase4_config, _config_signature(phase4_config))
                with st.status("Running Intelligent Refinement...", expanded=True) as status:
                    st.write("Generating negative prompt...")
                    result = service.run_intelligent_refinement(prompt.strip(), seed=int(seed))
                    status.update(label="Pipeline complete!", state="complete", expanded=False)

                st.subheader("Results")
                col1, col2 = st.columns(2)
                with col1:
                    if result.baseline_image_path:
                        st.image(Image.open(result.baseline_image_path), caption="Baseline", use_container_width=True)
                        st.metric("Baseline CLIP", f"{result.baseline_clip_score:.4f}" if result.baseline_clip_score else "N/A")
                with col2:
                    if result.winner_image_path:
                        st.image(Image.open(result.winner_image_path), caption=f"Winner ({result.winner_critic})", use_container_width=True)
                        st.metric("Winner CLIP", f"{result.winner_clip_score:.4f}" if result.winner_clip_score else "N/A")

                st.metric("CLIP Delta", f"{result.clip_score_delta:+.4f}" if result.clip_score_delta else "N/A")
                if result.negative_prompt:
                    st.caption(f"Negative: {result.negative_prompt}")
                st.caption(f"Total runtime: {result.total_runtime_seconds:.1f}s" if result.total_runtime_seconds else "")

                # Show per-critic scores
                if result.critic_results:
                    st.subheader("Critic Scoreboard")
                    cols = st.columns(len(result.critic_results))
                    for i, cr in enumerate(result.critic_results):
                        with cols[i]:
                            st.markdown(f"**{cr['critic'].title()}**")
                            st.metric("Best CLIP", f"{cr['best_clip_score']:.4f}" if cr['best_clip_score'] else "N/A")
                            st.caption(f"N={len(cr['candidates'])} images")

        except Exception as exc:
            st.error(f"Workflow failed: {exc}")


if __name__ == "__main__":
    main()
