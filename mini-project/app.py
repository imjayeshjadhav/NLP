"""
app.py
------
Streamlit front-end for the transformer summarization mini-project.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
import os

# Make sure the src package is importable whether the app is launched from
# the project root or anywhere else.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from src.preprocessing import SAMPLE_ARTICLE, SAMPLE_REFERENCE_SUMMARY, compare_cleaning
from src.summarizer import (
    generate_summary_bart,
    generate_summary_t5,
    generate_summary_fine_tuned,
    FINE_TUNED_DIR,
)
from src.evaluation import compute_rouge, build_comparison, plot_rouge_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")


# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Transformer Text Summarization",
    page_icon=None,
    layout="wide",
)

st.title("Transformer-Based Text Summarization")
st.caption(
    "Compare BART, T5 and a fine-tuned T5 on the same article, with ROUGE "
    "evaluation and side-by-side metrics."
)


# --------------------------------------------------------------------------- #
# Sidebar — generation controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Generation settings")

    model_choice = st.selectbox(
        "Model",
        options=["BART", "T5", "Fine-Tuned T5", "Compare All"],
        index=0,
        help="Pick a single model or compare all three.",
    )
    max_length = st.slider("max_length", 40, 300, 150, 10)
    min_length = st.slider("min_length", 10, 150, 40, 5)
    num_beams = st.slider("num_beams", 1, 8, 4, 1)

    st.divider()
    st.caption(
        "Fine-tuned model path:\n`{}`\n\n"
        "If missing, the fine-tuned option falls back to base T5. "
        "Run `python -m src.fine_tune` to create one.".format(FINE_TUNED_DIR)
    )


# --------------------------------------------------------------------------- #
# Main input area
# --------------------------------------------------------------------------- #
col_input, col_ref = st.columns([2, 1])

with col_input:
    st.subheader("Article")
    text = st.text_area(
        "Paste the article to summarize",
        value=SAMPLE_ARTICLE,
        height=260,
    )

with col_ref:
    st.subheader("Reference summary (optional)")
    reference = st.text_area(
        "Gold summary (used for ROUGE)",
        value=SAMPLE_REFERENCE_SUMMARY,
        height=260,
    )


# Show raw vs cleaned text for transparency
with st.expander("Show text cleaning (before vs after)"):
    info = compare_cleaning(text)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Before** — {info['before_length']} chars")
        st.code(info["before"][:1200] + ("…" if len(info["before"]) > 1200 else ""))
    with c2:
        st.markdown(f"**After** — {info['after_length']} chars")
        st.code(info["after"][:1200] + ("…" if len(info["after"]) > 1200 else ""))


# --------------------------------------------------------------------------- #
# Generate button
# --------------------------------------------------------------------------- #
generate = st.button("Generate Summary", type="primary", use_container_width=True)


def _run_model(key: str):
    """Dispatch helper with unified error handling."""
    try:
        if key == "BART":
            return generate_summary_bart(text, max_length, min_length, num_beams)
        if key == "T5":
            return generate_summary_t5(text, max_length, min_length, num_beams)
        if key == "Fine-Tuned T5":
            return generate_summary_fine_tuned(text, max_length, min_length, num_beams)
    except Exception as e:  # noqa: BLE001
        logger.exception("%s failed", key)
        st.error(f"{key} failed: {e}")
    return None


if generate:
    if not text.strip():
        st.warning("Please paste some text to summarize first.")
        st.stop()

    targets = (
        ["BART", "T5", "Fine-Tuned T5"] if model_choice == "Compare All" else [model_choice]
    )

    results = {}
    with st.spinner("Running models — first run will download weights…"):
        for t in targets:
            res = _run_model(t)
            if res is not None:
                results[t] = res

    if not results:
        st.stop()

    # ----- Display summaries -------------------------------------------------
    st.subheader("Summaries")
    for label, res in results.items():
        with st.container(border=True):
            st.markdown(f"### {label}")
            st.write(res.summary)
            meta_cols = st.columns(4)
            meta_cols[0].metric("Input words", res.input_length)
            meta_cols[1].metric("Summary words", res.summary_length)
            meta_cols[2].metric(
                "Compression",
                f"{(res.summary_length / res.input_length * 100):.1f}%"
                if res.input_length
                else "—",
            )
            meta_cols[3].metric("Time (s)", f"{res.inference_time_s:.2f}")

            if reference.strip():
                scores = compute_rouge(res.summary, reference)
                st.markdown("**ROUGE vs reference**")
                rcols = st.columns(3)
                rcols[0].metric("ROUGE-1", f"{scores['rouge1']:.3f}")
                rcols[1].metric("ROUGE-2", f"{scores['rouge2']:.3f}")
                rcols[2].metric("ROUGE-L", f"{scores['rougeL']:.3f}")

    # ----- Comparison table --------------------------------------------------
    if len(results) > 1:
        st.subheader("Comparison")
        summaries_map = {k: v.summary for k, v in results.items()}
        rows = build_comparison(summaries_map, reference=reference or None)
        st.dataframe(rows, use_container_width=True)

        if reference.strip():
            fig = plot_rouge_scores(rows)
            st.pyplot(fig)
