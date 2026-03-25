# RenAIssance Test II: Handwritten VLM OCR Pipeline

This repository contains my GSoC 2026 test submission for the **RenAIssance** project, focusing on **Test II: Text recognition with Vision-Language Model (VLM) integration** for 17th-century Spanish handwritten sources.

## Architecture & Approach

Unlike traditional OCR pipelines that rely on segmented line recognition and separate language models, this project implements an **End-to-End VLM Pipeline**. 

| Component | Technology | Description |
|---|---|---|
| **VLM Core** | `gemini-2.5-flash` | Natively processes high-resolution manuscript images. |
| **Paleography Logic** | Zero-Shot Prompting | Specialized system instructions to act as a 17th-century Spanish paleographer. |
| **Preprocessing** | PyMuPDF + PIL | Extracts raw PDF scans at 300 DPI and safely downscales to 3000px for API compliance. |
| **Evaluation** | Edit Distance (CER/WER) | Benchmarked against recovered DOCX ground-truth transcriptions. |

### Key Features
- **Zero-Shot Transcription**: Leverages the vision capabilities of Gemini to transcribe cursive script without training a specific CRNN/CTC model.
- **Paleographic Resolution**: Automatically expands historical abbreviations (e.g., `pedim.to` → `pedimento`) and resolves archaic ligatures.
- **Robustness**: Implements exponential backoff retries and idempotency (resume support) to handle transient API 503/429 errors.

## Performance Metrics

The pipeline achieved a consolidated **15.2% Character Error Rate (CER)** across 15 manuscript pages. 

The majority of "errors" are actually semantic improvements where the VLM correctly expands historical shorthand which the ground truth recorded literally.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Extract Images
```bash
python scripts/extract_pages.py --input-dir data/raw_pdfs --output-dir data/page_images
```

### 2. Parse Ground Truth
```bash
python scripts/parse_docx_ground_truth.py
```

### 3. Run VLM Inference
```bash
export GEMINI_API_KEY="your_api_key"
python scripts/vlm_extract.py
```

### 4. Evaluate
```bash
python scripts/evaluate_results.py \
  --predictions-file data/predictions/vlm_results.jsonl \
  --ground-truth-file data/ground_truth/ground_truth.jsonl \
  --output-dir data/predictions/eval_output
```

## Notebooks

- `notebooks/01_vlm_handwritten_ocr.ipynb` – The primary GSoC submission notebook. Demonstrates image extraction, zero-shot inference, and CER evaluation.

## Repository Map

- `src/vlm_pipeline.py` – Core Gemini VLM integration and resizing logic.
- `scripts/vlm_extract.py` – Batch execution script with retry logic.
- `report/gsoc_test_2_results.md` – Formal evaluation report and metric table.
