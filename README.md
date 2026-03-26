# RenAIssance Test II: Handwritten VLM OCR

Repo for my GSoC 2026 **RenAIssance** applicant test (**Test II**: handwritten text, VLM-based pipeline). I’m using **Gemini 2.5 Flash** on full page images instead of a line-by-line classical OCR stack.

## What’s in here

| Piece | What I used |
|-------|-------------|
| Model | `gemini-2.5-flash` (`google-genai`) |
| Prompting | Single paleography-style instruction in `src/vlm_pipeline.py` |
| PDFs → PNGs | PyMuPDF in `scripts/extract_pages.py`, 300 DPI; images are shrunk to max 3000px before the API call |
| Scoring | CER and WER vs `data/ground_truth/ground_truth.jsonl` (`src/evaluate.py`, `scripts/evaluate_results.py`) |

I also have older TrOCR/layout code under `src/` and other notebooks; **the numbers in the report** are from the VLM path only.

## Numbers (short version)

15 pages in the manifest get a VLM transcript. Only **5** of those have matching ground truth in this repo, so that’s what I average over: about **16.75% CER** and **40% WER** on those five. Details and per-page table: `report/gsoc_test_2_results.md`. Predictions frozen for reviewers: `data/predictions/vlm_results_submission.jsonl`.

CER is strict string edit distance. The model often spells out abbreviations the ground truth leaves shortened, so the score looks worse than the text “feels” when you read it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

**1. PDF → images + manifest**

```bash
python scripts/extract_pages.py --input-dir data/raw_pdfs --output-dir data/page_images
```

**2. Ground truth** (if you’re rebuilding it from docx)

```bash
python scripts/parse_docx_ground_truth.py
```

**3. VLM pass** (needs a key; burns quota)

```bash
export GEMINI_API_KEY="your_api_key"
python scripts/vlm_extract.py
```

**4. Evaluate** (submission snapshot if you have it)

```bash
python scripts/evaluate_results.py \
  --predictions-file data/predictions/vlm_results_submission.jsonl \
  --ground-truth-file data/ground_truth/ground_truth.jsonl \
  --output-dir data/predictions/eval_output
```

## Notebook

`notebooks/01_vlm_handwritten_ocr.ipynb` — what I’m sending with the application: load a page, load predictions, print aggregate CER/WER and one side-by-side excerpt.

## Main files

- `src/vlm_pipeline.py` — Gemini call + resize
- `scripts/vlm_extract.py` — batch over manifest, resume-friendly
- `report/gsoc_test_2_results.md` — write-up for Test II
