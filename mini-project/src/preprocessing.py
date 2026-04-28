"""
preprocessing.py
----------------
Text cleaning and normalization utilities for the summarization pipeline.

The cleaning steps are intentionally conservative so that the semantic
content of the article is preserved for the transformer models while
removing boilerplate that hurts summarization quality.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict

logger = logging.getLogger(__name__)

# Reference article used across the project when no user text is supplied.
SAMPLE_ARTICLE = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines, "
    "as opposed to the natural intelligence displayed by humans or animals. "
    "Leading AI textbooks define the field as the study of 'intelligent agents': "
    "any system that perceives its environment and takes actions that maximize "
    "its chance of achieving its goals. Some popular accounts use the term "
    "'artificial intelligence' to describe machines that mimic 'cognitive' "
    "functions that humans associate with the human mind, such as 'learning' "
    "and 'problem solving', however this definition is rejected by major AI "
    "researchers. AI applications include advanced web search engines (e.g., "
    "Google), recommendation systems (used by YouTube, Amazon and Netflix), "
    "understanding human speech (such as Siri and Alexa), self-driving cars "
    "(e.g., Tesla), automated decision-making and competing at the highest "
    "level in strategic game systems (such as chess and Go). As machines "
    "become increasingly capable, tasks considered to require 'intelligence' "
    "are often removed from the definition of AI, a phenomenon known as the "
    "AI effect. For instance, optical character recognition is frequently "
    "excluded from things considered to be AI, having become a routine "
    "technology. Modern machine learning techniques are at the core of AI. "
    "Problems for AI applications include reasoning, knowledge representation, "
    "planning, learning, natural language processing, perception, and the "
    "ability to move and manipulate objects."
)

# Short reference summary used for ROUGE evaluation against the sample.
SAMPLE_REFERENCE_SUMMARY = (
    "Artificial intelligence is intelligence shown by machines that perceive "
    "their environment and act to achieve goals. AI powers search engines, "
    "recommendation systems, voice assistants and self-driving cars, and its "
    "core problems include reasoning, learning, planning and natural "
    "language processing."
)


def _strip_control_chars(text: str) -> str:
    """Remove non-printable / control characters that survive unicode NFKC."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def clean_text(text: str) -> str:
    """
    Clean and normalize raw article text.

    Steps:
        1. Unicode normalization (NFKC) — collapses weird width / ligatures.
        2. Strip control characters.
        3. Remove URLs and bracketed citations like "[1]" that appear in Wiki text.
        4. Keep alphanumerics, basic punctuation and whitespace.
        5. Collapse repeated whitespace into a single space.

    Args:
        text: raw input string.

    Returns:
        Cleaned, single-spaced string safe to feed into a tokenizer.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects str, got {type(text).__name__}")

    if not text.strip():
        return ""

    # 1. Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    # 2. Strip control chars
    text = _strip_control_chars(text)
    # 3. Drop URLs and bracket citations
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    # 4. Keep sensible characters only
    text = re.sub(r"[^A-Za-z0-9.,;:!?'\"()\-\s]", " ", text)
    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_for_model(text: str, model_name: str = "bart") -> str:
    """
    Apply model-specific prefixes on top of the generic cleaning.

    T5 expects an explicit task prefix ("summarize: ..."); BART does not.
    """
    cleaned = clean_text(text)
    if model_name.lower().startswith("t5"):
        return "summarize: " + cleaned
    return cleaned


def compare_cleaning(text: str) -> Dict[str, str]:
    """Return a side-by-side dict of the raw vs cleaned text for display."""
    cleaned = clean_text(text)
    return {
        "before": text,
        "after": cleaned,
        "before_length": str(len(text)),
        "after_length": str(len(cleaned)),
    }


if __name__ == "__main__":
    # Quick manual smoke test
    logging.basicConfig(level=logging.INFO)
    dirty = "  Hello   world!! \n Visit https://example.com [1]  — it's great.  "
    print("BEFORE:", repr(dirty))
    print("AFTER :", repr(clean_text(dirty)))
