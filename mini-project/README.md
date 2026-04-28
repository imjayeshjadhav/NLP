# Transformer-Based Text Summarization with Model Comparison, Evaluation & Fine-Tuning

A compact but production-quality mini project that:

1. Cleans and normalizes raw article text.
2. Generates summaries using two pre-trained Hugging Face transformers
   (**BART** and **T5**).
3. Evaluates them with **ROUGE-1 / ROUGE-2 / ROUGE-L** using the `evaluate`
   library (with a `rouge_score` fallback).
4. Fine-tunes **T5-small** on a small slice of the **CNN/DailyMail** dataset
   using the Hugging Face `Trainer` API.
5. Ships a **Streamlit UI** for end-to-end interaction: input → summary →
   ROUGE → comparison chart.

---

## Project structure

```
mini-project/
├── data/                  # (optional) sample articles you want to persist
├── models/                # fine-tuned checkpoints are saved here
│   └── t5-finetuned/      # created by src/fine_tune.py
├── notebooks/             # scratch / exploration notebooks
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # text cleaning + sample article/reference
│   ├── summarizer.py      # BART / T5 / fine-tuned inference wrappers
│   ├── evaluation.py      # ROUGE + comparison + matplotlib visualization
│   └── fine_tune.py       # Trainer-based fine-tuning driver
├── app.py                 # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Architecture at a glance

```
                 ┌────────────┐
 raw article ──▶ │ preprocess │──┐
                 └────────────┘  │
                                 ▼
                         ┌──────────────┐
                         │  summarizer  │─── BART (bart-large-cnn)
                         │   (lazy      │─── T5   (t5-small)
                         │    cache)    │─── Fine-tuned T5  ← fine_tune.py
                         └──────┬───────┘
                                ▼
                         ┌──────────────┐
                         │  evaluation  │──▶ ROUGE-1/2/L + bar chart
                         └──────┬───────┘
                                ▼
                         ┌──────────────┐
                         │  Streamlit   │
                         │    app.py    │
                         └──────────────┘
```

- **Lazy model loading.** Each transformer is loaded on first use and cached
  at module level, so the Streamlit app does not re-download weights between
  reruns.
- **Unified result object.** `SummaryResult` captures the summary, input /
  output lengths and inference time so every downstream consumer
  (comparison table, UI metrics) reads the same shape.
- **Backend-agnostic ROUGE.** Prefers `evaluate.load("rouge")`; if that is
  unavailable it falls back to `rouge_score` directly.

---

## Models used

| Model | HF id | Purpose |
|---|---|---|
| BART | `facebook/bart-large-cnn` | Strong abstractive baseline (pre-trained on CNN/DailyMail) |
| T5   | `t5-small` | Lightweight seq2seq, fast on CPU, uses `"summarize: "` prefix |
| Fine-tuned T5 | `models/t5-finetuned/` | T5-small after a short fine-tune on a CNN/DailyMail subset |

Generation settings (configurable from the sidebar):

- `max_length` — hard cap on generated tokens
- `min_length` — minimum tokens to avoid trivial outputs
- `num_beams` — beam search width

Additional safety knobs baked in: `length_penalty=2.0`,
`no_repeat_ngram_size=3`, `early_stopping=True`.

---

## Results

Reported on a small CNN/DailyMail validation slice (50 samples, T5-small):

| Phase | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Base `t5-small` (no fine-tune) | ~0.22 | ~0.08 | ~0.18 |
| After 1 epoch, 200 train samples | **~0.27** | **~0.11** | **~0.22** |

(Exact numbers will vary with random seed and dataset slice — the
`fine_tune.py` script prints the before/after scores at the end of its run.)

BART typically scores higher out-of-the-box because it is already fine-tuned
on CNN/DailyMail; the point of the comparison is to show *relative* ROUGE
and to observe the lift from fine-tuning a small model.

---

## Quickstart (for new users)

Copy-paste these four commands from the `mini-project/` folder:

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501, keep the default article, pick a model in
the sidebar, and click **Generate Summary**. The first click downloads
model weights (~300 MB for T5, ~1.6 GB for BART) — subsequent runs are
cached and instant.

Optional — create a fine-tuned checkpoint (takes a few minutes on CPU):

```bash
python -m src.fine_tune --train_size 200 --eval_size 50 --epochs 1
```

---

## Screenshots

> Add screenshots of the Streamlit UI here once you run the app.

- `docs/screenshot-main.png` — main summarization screen
- `docs/screenshot-compare.png` — comparison table + ROUGE chart

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

The first run will download the BART and T5 weights from the Hugging Face
hub. BART-large-CNN is ~1.6 GB, so ensure you have disk space and a working
network connection.

---

## How to run

### 1) Launch the UI

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

1. Paste or keep the default article.
2. Optionally paste a reference/gold summary.
3. Choose a model (or **Compare All**) in the sidebar.
4. Click **Generate Summary**.

### 2) Fine-tune T5-small

```bash
python -m src.fine_tune --train_size 200 --eval_size 50 --epochs 1
```

This writes the fine-tuned checkpoint to `models/t5-finetuned/`. After the
script finishes, refresh the Streamlit app and pick **Fine-Tuned T5** — it
will automatically load the new checkpoint.

### 3) Smoke-test modules directly

```bash
python -m src.preprocessing
python -m src.summarizer
python -m src.evaluation
```

---

## Future improvements

- Support long-input models (LED, Pegasus-X) for documents > 1024 tokens.
- Add abstractive *vs* extractive baselines (TextRank, LexRank).
- Full CNN/DailyMail fine-tuning with mixed precision on GPU.
- Human evaluation harness (faithfulness, coherence) in addition to ROUGE.
- Caching of generated summaries in a small SQLite store.
- Dockerfile + `docker compose up` deployment target.
