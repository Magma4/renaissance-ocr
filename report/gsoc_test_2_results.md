# Test II write-up — handwritten VLM OCR

GSoC 2026, RenAIssance applicant test (handwritten sources).

## What the test asked for

Test II wants a pipeline built around an LLM or VLM, with the model doing real work on the image and transcript—not only cleaning up output from a separate OCR engine. Here, **Gemini 2.5 Flash** takes the page image (after resizing in `vlm_pipeline.py`) and returns the transcription in one shot. There is TrOCR/layout code elsewhere in the repo, but **the scores below are VLM-only** on the pages that have ground truth.

Metrics: **CER** and **WER** after the same whitespace normalization for hypothesis and reference (`normalize_for_metrics` in `src/utils.py`, Levenshtein in `src/evaluate.py`).

## Data

- **15** pages in `manifest.jsonl` → I ran the VLM on all of them (`vlm_results_submission.jsonl`).
- **5** pages have text in `ground_truth.jsonl` (one page per bundle, same `page_id` as the manifest).
- The other **10** transcripts aren’t scored here because there’s no reference string for them in-repo.

I kept a copy of the predictions I report on as `data/predictions/vlm_results_submission.jsonl`. If you change `vlm_extract.py` output, re-run evaluation; numbers drift a bit run-to-run anyway.

## Overall scores (those 5 pages)

| | |
|--|--|
| Mean CER | 16.75% (0.1675) |
| Mean WER | 40.19% (0.4019) |

## Per page

Same evaluation run as the averages above.

| Source | CER | WER |
|--------|-----|-----|
| AHPG-GPAH 1:1716 | 0.1881 | 0.4151 |
| AHPG-GPAH AU61:2 | 0.1913 | 0.4103 |
| AHN Inquisición 1667 | 0.1840 | 0.4641 |
| PT3279:146:342 (1857) | 0.0946 | 0.2791 |
| Pleito Marqués de Viana | 0.1796 | 0.4408 |

## How to reproduce

```bash
source .venv/bin/activate   # optional

python scripts/evaluate_results.py \
  --predictions-file data/predictions/vlm_results_submission.jsonl \
  --ground-truth-file data/ground_truth/ground_truth.jsonl \
  --output-dir data/predictions/eval_output
```

`summary.json` / `comparisons.jsonl` land in `eval_output/`. The submission notebook calls the same `evaluate_records` helper and prints a short excerpt for one page.

To redo inference (optional):

```bash
export GEMINI_API_KEY="…"
python scripts/vlm_extract.py
```

## Caveats

Ground truth is **literal** (abbreviations, tildes, line breaks as typed). The model often **expands** or normalizes; that’s useful for reading but hurts CER. There are also straight misreads on names and place names—you can see both in the notebook excerpt.
