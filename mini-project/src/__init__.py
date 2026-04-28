"""
Transformer-Based Text Summarization package.

Modules:
    preprocessing : text cleaning and normalization utilities
    summarizer    : BART / T5 / fine-tuned summary generation
    evaluation    : ROUGE metric computation and visualization
    fine_tune     : lightweight fine-tuning driver for T5 on CNN/DailyMail
"""

__all__ = ["preprocessing", "summarizer", "evaluation", "fine_tune"]
__version__ = "1.0.0"
