"""
summarizer.py
-------------
Wraps Hugging Face BART and T5 models behind a small, cached interface so
the rest of the project (and the Streamlit UI) can generate summaries
without worrying about model loading or device placement.

Design notes
~~~~~~~~~~~~
* Models are loaded lazily and cached in module-level dicts to avoid
  re-downloading / re-loading on every call (important for Streamlit).
* A single ``generate_summary`` entry point dispatches on a model key so
  the UI only needs to pass a string ("bart", "t5", "fine-tuned").
* Inference timing is returned alongside the summary for the comparison
  table.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from .preprocessing import clean_text, preprocess_for_model

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Model identifiers
# --------------------------------------------------------------------------- #
BART_MODEL_NAME = "facebook/bart-large-cnn"
# NOTE: we use flan-t5-small instead of plain t5-small because the latter
# only ships a SentencePiece blob and requires `sentencepiece` at runtime,
# which has flaky wheels on Python 3.13. flan-t5-small ships a full
# tokenizer.json so the fast tokenizer loads cleanly.
T5_MODEL_NAME = "google/flan-t5-small"

# Location where the fine-tuned model is saved by fine_tune.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINE_TUNED_DIR = os.path.join(_PROJECT_ROOT, "models", "t5-finetuned")

# Single source of truth for the device used by all models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy caches: (tokenizer, model)
_cache: Dict[str, tuple] = {}


@dataclass
class SummaryResult:
    """Container returned by every ``generate_summary_*`` call."""

    model_name: str
    summary: str
    input_length: int
    summary_length: int
    inference_time_s: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "model": self.model_name,
            "summary": self.summary,
            "input_length": self.input_length,
            "summary_length": self.summary_length,
            "compression_ratio": (
                round(self.summary_length / self.input_length, 3)
                if self.input_length
                else 0.0
            ),
            "inference_time_s": round(self.inference_time_s, 3),
        }


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_bart():
    """Load BART-large-CNN (cached). Uses AutoTokenizer so the fast
    (tokenizers-backed) variant is preferred — no sentencepiece needed."""
    if "bart" not in _cache:
        logger.info("Loading BART model: %s", BART_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        _cache["bart"] = (tokenizer, model)
    return _cache["bart"]


def _load_t5():
    """Load T5-small (cached). AutoTokenizer picks T5TokenizerFast which
    reads tokenizer.json directly and does NOT need sentencepiece."""
    if "t5" not in _cache:
        logger.info("Loading T5 model: %s", T5_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        _cache["t5"] = (tokenizer, model)
    return _cache["t5"]


def _has_checkpoint_files(path: str) -> bool:
    """True iff the directory looks like a saved HF model (config + weights)."""
    if not os.path.isdir(path):
        return False
    entries = set(os.listdir(path))
    has_config = "config.json" in entries
    has_weights = any(
        f.endswith((".safetensors", ".bin")) for f in entries
    )
    return has_config and has_weights


def _load_fine_tuned():
    """Load a locally fine-tuned model if it exists, else fall back to T5."""
    if "fine_tuned" in _cache:
        return _cache["fine_tuned"]

    if not _has_checkpoint_files(FINE_TUNED_DIR):
        logger.warning(
            "No fine-tuned model found at %s — falling back to base T5. "
            "Run `python -m src.fine_tune` first to create one.",
            FINE_TUNED_DIR,
        )
        return _load_t5()

    logger.info("Loading fine-tuned model from %s", FINE_TUNED_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_DIR)
    model.to(DEVICE)
    model.eval()
    _cache["fine_tuned"] = (tokenizer, model)
    return _cache["fine_tuned"]


# --------------------------------------------------------------------------- #
# Core generation helper
# --------------------------------------------------------------------------- #
def _generate(
    tokenizer,
    model,
    text: str,
    model_label: str,
    *,
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 4,
    max_input_tokens: int = 1024,
) -> SummaryResult:
    """
    Shared generation routine. Handles tokenization, beam search and timing.
    """
    if not text or not text.strip():
        raise ValueError("Cannot summarize empty text.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_input_tokens,
        truncation=True,
    ).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    elapsed = time.perf_counter() - start

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return SummaryResult(
        model_name=model_label,
        summary=summary,
        input_length=len(text.split()),
        summary_length=len(summary.split()),
        inference_time_s=elapsed,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_summary_bart(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 4,
) -> SummaryResult:
    """Summarize ``text`` using facebook/bart-large-cnn."""
    try:
        cleaned = clean_text(text)
        tokenizer, model = _load_bart()
        return _generate(
            tokenizer,
            model,
            cleaned,
            "BART (facebook/bart-large-cnn)",
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
        )
    except Exception as e:
        logger.exception("BART summarization failed: %s", e)
        raise


def generate_summary_t5(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 4,
) -> SummaryResult:
    """Summarize ``text`` using t5-small with the 'summarize:' prefix."""
    try:
        prepared = preprocess_for_model(text, model_name="t5")
        tokenizer, model = _load_t5()
        return _generate(
            tokenizer,
            model,
            prepared,
            "T5 (t5-small)",
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            max_input_tokens=512,  # t5-small max encoder length
        )
    except Exception as e:
        logger.exception("T5 summarization failed: %s", e)
        raise


def generate_summary_fine_tuned(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 4,
) -> SummaryResult:
    """Summarize using the fine-tuned checkpoint, if available."""
    try:
        prepared = preprocess_for_model(text, model_name="t5")
        tokenizer, model = _load_fine_tuned()
        return _generate(
            tokenizer,
            model,
            prepared,
            "Fine-Tuned T5",
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            max_input_tokens=512,
        )
    except Exception as e:
        logger.exception("Fine-tuned summarization failed: %s", e)
        raise


def generate_summary(
    text: str,
    model_key: str = "bart",
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 4,
) -> SummaryResult:
    """
    Dispatch helper used by the Streamlit UI.

    ``model_key`` must be one of: "bart", "t5", "fine-tuned".
    """
    key = model_key.lower().strip()
    if key == "bart":
        return generate_summary_bart(text, max_length, min_length, num_beams)
    if key == "t5":
        return generate_summary_t5(text, max_length, min_length, num_beams)
    if key in {"fine-tuned", "fine_tuned", "finetuned"}:
        return generate_summary_fine_tuned(text, max_length, min_length, num_beams)
    raise ValueError(
        f"Unknown model_key '{model_key}'. Expected one of: bart, t5, fine-tuned."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .preprocessing import SAMPLE_ARTICLE

    print("Running on device:", DEVICE)
    bart_out = generate_summary_bart(SAMPLE_ARTICLE)
    t5_out = generate_summary_t5(SAMPLE_ARTICLE)
    print("\n--- BART ---")
    print(bart_out.as_dict())
    print("\n--- T5 ---")
    print(t5_out.as_dict())
