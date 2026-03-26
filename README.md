# RenAIssance Test I: Printed OCR Baseline

This repo is my GSoC 2026 test submission for the RenAIssance project (printed OCR track).
The goal is to recognise main text from scanned 17th-century Spanish printed sources.

## Architecture & Approach

The pipeline implements two OCR backends:

| Backend | Architecture | Description |
|---|---|---|
| **CRNN** (primary) | CNN-RNN + CTC | 4-block CNN backbone → BiLSTM → CTC decode |
| **TrOCR** (baseline) | Vision Transformer | `microsoft/trocr-base-printed`, zero-shot or fine-tuned |

Both share the same preprocessing and layout pipeline. The CRNN backend is the core deliverable for the GSoC task; TrOCR is kept as a strong zero-shot baseline for comparison.

### Weighted Learning for Rare Characters

The CRNN training script computes **per-character inverse-frequency weights** over the ground-truth corpus.
Rare letterforms (e.g. `ñ`, `ü`, ligatures) receive proportionally higher weights in the CTC loss, which forces the model to attend to characters that would otherwise be swamped by high-frequency common letters.

### Constrained Beam Search with a Renaissance Spanish Lexicon

The CRNN decoder offers two decoding modes:

- **Greedy** – argmax collapse, fastest.
- **Beam search** (`--crnn-decoder beam`) – prefix beam search with lexicon rescoring.  
  A curated wordlist of ~250 period-appropriate Spanish words (`data/lexicon/spanish_renaissance_lexicon.txt`) provides a small bonus to completed words at word boundaries, nudging the decoder toward attested orthography without blocking OOV tokens.

### LLM Integration (Gemini)

The post-OCR cleanup step now supports three backends:

- `rule_based` – fully local regex cleanup.
- `manual` – writes prompts to disk; reads back saved responses.
- `gemini` – sends raw OCR output to the Gemini API with a historically-informed prompt that preserves period spelling while correcting obvious recognition errors.

## Pipeline

```
PDF → page images → preprocessing → main-text crop → line segmentation
    → CRNN or TrOCR (line-by-line OCR)
    → Gemini / rule-based cleanup
    → CER / WER evaluation (before and after cleanup)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Pipeline

### Step 0 – Extract page images

```bash
python scripts/extract_pages.py \
  --input-dir data/raw_pdfs \
  --output-dir data/page_images \
  --dpi 300
```

### Step 0b – Build ground truth JSONL from transcription

```bash
python scripts/parse_ground_truth.py \
  --input-file data/ground_truth/transcription_marked.txt \
  --output-file data/ground_truth/ground_truth.jsonl
```

### Step 1 – Run OCR baseline (TrOCR, zero-shot)

```bash
python scripts/run_baseline.py \
  --input-dir data/page_images \
  --predictions-file data/predictions/baseline_predictions.jsonl \
  --cleanup-backend rule_based
```

### Step 1b – Run OCR with Gemini cleanup

```bash
GEMINI_API_KEY=your_key python scripts/run_baseline.py \
  --input-dir data/page_images \
  --predictions-file data/predictions/gemini_predictions.jsonl \
  --cleanup-backend gemini
```

### Step 2 – Train the CRNN model

```bash
python scripts/train_crnn.py \
  --ground-truth-file data/ground_truth/ground_truth.jsonl \
  --line-image-dir data/crops \
  --output-dir outputs/crnn \
  --epochs 30
```

### Step 2b – Fine-tune TrOCR on ground truth

```bash
python scripts/finetune_trocr.py \
  --ground-truth-file data/ground_truth/ground_truth.jsonl \
  --image-dir data/crops \
  --output-dir outputs/trocr_finetuned \
  --epochs 5
```

### Step 3 – Run OCR with trained CRNN backend

```bash
# Greedy decoding
python scripts/run_baseline.py \
  --input-dir data/page_images \
  --predictions-file data/predictions/crnn_predictions.jsonl \
  --model-backend crnn \
  --crnn-checkpoint outputs/crnn/crnn_checkpoint.pt \
  --crnn-vocab outputs/crnn/crnn_vocab.json \
  --cleanup-backend gemini

# Lexicon beam search
python scripts/run_baseline.py \
  --input-dir data/page_images \
  --predictions-file data/predictions/crnn_beam_predictions.jsonl \
  --model-backend crnn \
  --crnn-checkpoint outputs/crnn/crnn_checkpoint.pt \
  --crnn-vocab outputs/crnn/crnn_vocab.json \
  --crnn-decoder beam \
  --lexicon-path data/lexicon/spanish_renaissance_lexicon.txt
```

### Step 4 – Evaluate

```bash
python scripts/evaluate_results.py \
  --predictions-file data/predictions/baseline_predictions.jsonl \
  --ground-truth-file data/ground_truth/ground_truth.jsonl \
  --output-dir outputs/sample_predictions
```

## Evaluation Metrics

**Character Error Rate (CER)** and **Word Error Rate (WER)** are computed using edit distance (Levenshtein) between the normalised hypothesis and ground truth.  Text is normalised before scoring: line breaks are collapsed, repeated whitespace is merged — so scores are not inflated by formatting differences.

Both metrics are computed **before and after cleanup**. The delta `cer_delta = cleaned_cer - raw_cer` shows the contribution of the LLM correction step.

Test I numbers, methods, and reproduction commands are written up in `report/gsoc_test_1_results.md`.

## Notebooks

| Notebook | Contents |
|---|---|
| `01_data_exploration.ipynb` | Page image inspection |
| `02_baseline_ocr.ipynb` | TrOCR baseline + crop visualisation |
| `03_llm_cleanup_and_eval.ipynb` | Rule-based cleanup + CER/WER |
| `04_crnn_training.ipynb` | CRNN architecture, char weights, training |
| `05_gemini_cleanup.ipynb` | Gemini API integration, side-by-side CER comparison |

## Data Layout

- `data/raw_pdfs/` – input PDF scans
- `data/page_images/` – extracted PNG images + `manifest.jsonl`
- `data/ground_truth/transcription_marked.txt` – editable ground truth
- `data/ground_truth/ground_truth.jsonl` – parsed ground truth (generated)
- `data/crops/` – main-text crop images written by `run_baseline.py`
- `data/lexicon/spanish_renaissance_lexicon.txt` – period Spanish wordlist
- `data/predictions/` – JSONL prediction files
- `outputs/crnn/` – trained CRNN checkpoint + vocabulary
- `outputs/trocr_finetuned/` – fine-tuned TrOCR model
- `outputs/sample_predictions/` – evaluation summaries and examples

## Source Map

| File | Role |
|---|---|
| `src/preprocessing.py` | CLAHE + adaptive thresholding |
| `src/layout.py` | Projection-based main-text crop + line segmentation |
| `src/ocr_model.py` | TrOCR wrapper |
| `src/crnn_model.py` | CRNN model, weighted CTC, vocabulary helpers |
| `src/decoding.py` | Greedy + lexicon beam search CTC decoders |
| `src/postprocess_llm.py` | Rule-based, manual, and Gemini cleanup |
| `src/evaluate.py` | CER/WER, comparison artifacts |
| `src/utils.py` | Config dataclasses, I/O helpers |

## Limitations

- Ground truth is limited to a few transcribed pages per source; CRNN training benefits from more data.
- The layout heuristic will struggle on multi-column or heavily annotated pages.
- TrOCR is not natively fine-tuned on Spanish text; the CRNN provides a domain-specific learnable alternative.
