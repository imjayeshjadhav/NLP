"""
evaluation.py
-------------
ROUGE-based evaluation utilities plus a small comparison / visualization layer.

We prefer the `evaluate` library (which wraps Google's rouge_score) but fall
back to a direct `rouge_score` import if `evaluate` is not installed, so the
module still works in minimal environments.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# ROUGE backend loading (with graceful fallback)
# --------------------------------------------------------------------------- #
_rouge_metric = None
_rouge_scorer = None

try:
    import evaluate  # type: ignore

    _rouge_metric = evaluate.load("rouge")
    logger.info("Using `evaluate` library for ROUGE.")
except Exception as e:  # pragma: no cover - network / install issues
    logger.warning("`evaluate` backend unavailable (%s); falling back to rouge_score.", e)
    try:
        from rouge_score import rouge_scorer  # type: ignore

        _rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    except Exception as e2:  # pragma: no cover
        logger.error("No ROUGE backend available: %s", e2)


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-1 / ROUGE-2 / ROUGE-L F-measure between a predicted
    summary and a reference summary.

    Returns a dict such as:
        {"rouge1": 0.42, "rouge2": 0.18, "rougeL": 0.35}
    """
    if not prediction or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    if _rouge_metric is not None:
        results = _rouge_metric.compute(
            predictions=[prediction],
            references=[reference],
            use_stemmer=True,
        )
        # `evaluate` returns floats directly in recent versions
        return {
            "rouge1": float(results["rouge1"]),
            "rouge2": float(results["rouge2"]),
            "rougeL": float(results["rougeL"]),
        }

    if _rouge_scorer is not None:
        scores = _rouge_scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    logger.error("ROUGE requested but no backend is installed.")
    return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_rouge_batch(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Corpus-level ROUGE (mean F1 across the list)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be same length.")
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    if _rouge_metric is not None:
        results = _rouge_metric.compute(
            predictions=predictions, references=references, use_stemmer=True
        )
        return {k: float(results[k]) for k in ("rouge1", "rouge2", "rougeL")}

    # Fallback: average per-sample
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(predictions, references):
        s = compute_rouge(p, r)
        for k in agg:
            agg[k] += s[k]
    n = len(predictions)
    return {k: v / n for k, v in agg.items()}


# --------------------------------------------------------------------------- #
# Comparison logic
# --------------------------------------------------------------------------- #
def _redundancy_score(summary: str) -> float:
    """
    Approximate "redundancy" as 1 - (unique_tokens / total_tokens).
    Lower is better (less repetition).
    """
    tokens = summary.lower().split()
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def _clarity_score(summary: str) -> float:
    """
    Crude clarity proxy: favors summaries that are neither too short nor
    too repetitive. Returns a value in [0, 1].
    """
    n = len(summary.split())
    if n == 0:
        return 0.0
    # A bell-ish curve peaking around 60 tokens
    length_component = max(0.0, 1.0 - abs(n - 60) / 100.0)
    redundancy_component = 1.0 - _redundancy_score(summary)
    return round(0.5 * length_component + 0.5 * redundancy_component, 3)


def build_comparison(
    summaries: Dict[str, str],
    reference: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    Build a list of rows for a comparison table.

    Args:
        summaries: mapping of model label -> generated summary
        reference: optional gold summary. When provided, ROUGE scores are
                   added to each row.
    """
    rows = []
    for model_label, summary in summaries.items():
        row: Dict[str, object] = {
            "model": model_label,
            "length_words": len(summary.split()),
            "clarity": _clarity_score(summary),
            "redundancy": round(_redundancy_score(summary), 3),
        }
        if reference:
            row.update({k: round(v, 4) for k, v in compute_rouge(summary, reference).items()})
        rows.append(row)
    return rows


def print_comparison_table(rows: List[Dict[str, object]]) -> str:
    """
    Render a simple fixed-width comparison table and return it as a string.
    Also prints it to stdout for CLI use.
    """
    if not rows:
        return "(no rows)"

    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    def fmt_row(values):
        return " | ".join(str(v).ljust(widths[h]) for h, v in zip(headers, values))

    line = "-+-".join("-" * widths[h] for h in headers)
    out_lines = [
        fmt_row(headers),
        line,
        *[fmt_row([r[h] for h in headers]) for r in rows],
    ]
    table = "\n".join(out_lines)
    print(table)
    return table


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def plot_rouge_scores(rows: List[Dict[str, object]], save_path: Optional[str] = None):
    """
    Draw a grouped bar chart comparing ROUGE-1/2/L across models.
    Returns the matplotlib Figure so Streamlit can render it with st.pyplot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["rouge1", "rouge2", "rougeL"]
    models = [r["model"] for r in rows]
    data = np.array(
        [[float(r.get(m, 0.0)) for m in metrics] for r in rows]
    )

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, data[:, i], width, label=metric.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("ROUGE comparison")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        logger.info("Saved ROUGE plot to %s", save_path)
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pred = "The cat sat on the mat quietly."
    ref = "A cat was sitting quietly on the mat."
    print(compute_rouge(pred, ref))
