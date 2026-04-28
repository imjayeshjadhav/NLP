"""
fine_tune.py
------------
Light-weight fine-tuning driver for T5-small on a small slice of the
CNN/DailyMail dataset, using the Hugging Face ``Trainer`` API.

Design choices
~~~~~~~~~~~~~~
* Uses a *tiny* training subset (default 200 examples) so the script can
  finish on a CPU in a few minutes — this is a teaching project, not a
  production training run.
* Saves the result to ``<project>/models/t5-finetuned`` so that
  ``summarizer.generate_summary_fine_tuned`` can pick it up automatically.
* Reports ROUGE before vs after fine-tuning on a small eval set so the
  learning effect is visible in the logs and README.

Run with:
    python -m src.fine_tune --train_size 200 --eval_size 50 --epochs 1
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "models", "t5-finetuned")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "models", "checkpoints")

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
PREFIX = "summarize: "


def _load_dataset(train_size: int, eval_size: int):
    """Load a small slice of CNN/DailyMail."""
    from datasets import load_dataset

    logger.info("Loading CNN/DailyMail (3.0.0) — train[:%d], validation[:%d]",
                train_size, eval_size)
    train = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{train_size}]")
    val = load_dataset("cnn_dailymail", "3.0.0", split=f"validation[:{eval_size}]")
    return train, val


def _preprocess_factory(tokenizer):
    """Build a preprocess_function closed over the tokenizer."""

    def preprocess_function(examples):
        inputs = [PREFIX + doc for doc in examples["article"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=examples["highlights"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def _build_compute_metrics(tokenizer):
    """Build a compute_metrics callback that returns ROUGE F1 scores."""
    from .evaluation import compute_rouge_batch

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # If predictions are logits (3-D), take argmax to get token IDs
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # Clip to valid token ID range (removes any -100 or negative artifacts)
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)

        # Replace -100 in the labels (pad id used during training)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = compute_rouge_batch(decoded_preds, decoded_labels)
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def _evaluate_baseline(model, tokenizer, eval_dataset) -> Dict[str, float]:
    """Compute ROUGE on the eval set *before* fine-tuning (for comparison)."""
    import torch
    from .evaluation import compute_rouge_batch

    model.eval()
    device = next(model.parameters()).device

    preds, refs = [], []
    for ex in eval_dataset:
        inputs = tokenizer(
            PREFIX + ex["article"],
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
        refs.append(ex["highlights"])

    return compute_rouge_batch(preds, refs)


def fine_tune(
    train_size: int = 200,
    eval_size: int = 50,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
) -> Dict[str, Dict[str, float]]:
    """
    Run a tiny fine-tuning job and return before/after ROUGE scores.
    """
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_raw, val_raw = _load_dataset(train_size, eval_size)

    preprocess_function = _preprocess_factory(tokenizer)
    train_tok = train_raw.map(
        preprocess_function, batched=True, remove_columns=train_raw.column_names
    )
    val_tok = val_raw.map(
        preprocess_function, batched=True, remove_columns=val_raw.column_names
    )

    # --- Baseline ROUGE (before fine-tuning) ---------------------------------
    logger.info("Computing baseline ROUGE on %d eval samples…", len(val_raw))
    baseline_scores = _evaluate_baseline(model, tokenizer, val_raw)
    logger.info("Baseline ROUGE: %s", baseline_scores)

    # --- Trainer setup -------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_total_limit=1,
        predict_with_generate=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_build_compute_metrics(tokenizer),
    )

    logger.info("Starting fine-tuning: %d epochs, %d train / %d val",
                epochs, len(train_tok), len(val_tok))
    trainer.train()

    # --- Save final model ----------------------------------------------------
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Saved fine-tuned model to %s", OUTPUT_DIR)

    # --- Post fine-tuning ROUGE ---------------------------------------------
    after_scores = _evaluate_baseline(model, tokenizer, val_raw)
    logger.info("After fine-tuning ROUGE: %s", after_scores)

    return {"before": baseline_scores, "after": after_scores}


def _parse_args():
    p = argparse.ArgumentParser(description="Fine-tune T5-small on CNN/DailyMail (tiny slice)")
    p.add_argument("--train_size", type=int, default=200)
    p.add_argument("--eval_size", type=int, default=50)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    results = fine_tune(
        train_size=args.train_size,
        eval_size=args.eval_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print("\n=== ROUGE: before vs after fine-tuning ===")
    for phase, scores in results.items():
        print(f"{phase:>6}: {scores}")
