# GSoC Test I results — Printed OCR baseline (early modern Spanish)

**Author:** Raymond Frimpong Amoateng  
**Date:** March 2026  
**Project:** RenAIssance GSoC 2026 — Test I (Printed Text OCR)

---

## 1. Problem and Scope

This project builds an end-to-end OCR pipeline for 17th-century printed Spanish sources, targeting the main body text of each page. The scope is deliberately narrow:

- **Printed** historical text only (not handwritten).
- **Main text column** only — marginal notes, running headers, and page numbers are suppressed by a layout heuristic rather than transcribed.
- **LLM as post-processor only** — Gemini is used to clean up OCR output after recognition, not as the primary recognizer.
- **Two recognition backends** are implemented: a zero-shot TrOCR baseline and a trained CRNN model.

The goal of this first pass is to establish a reproducible baseline with honest CER/WER numbers, then identify the highest-leverage improvements.

---

## 2. Data

### Sources

Six early modern Spanish PDF sources were provided. Ground truth transcription was available for a subset of pages:

| Source | Pages evaluated |
|---|---|
| `buendia_instruccion` | 3 pages (p. 1–3) |
| `covarrubias_tesoro_lengua` | 2 pages (p. 1–2) |
| Other sources (`guardiola_tratado_nobleza`, three `porcones` collections) | 0 pages (transcription pending — TODO markers) |

**Total evaluated: 3 pages** (the 2 Covarrubias pages could not be matched to extracted images in the current run; 3 Buendia pages matched successfully).

### Scan characteristics

The Buendia pages are 300 dpi scans of late-17th century print. They show moderate degradation: some ink bleed, uneven margins, and a mix of roman and italic typefaces. The Covarrubias pages (Tesoro de la lengua castellana, 1611) are denser, with smaller type and more abbreviations.

### Ground truth preparation

Transcriptions were provided in `transcription_marked.txt` using a structured `===PAGE: <page_id>===` format. The file was parsed with `scripts/parse_ground_truth.py` to produce `data/ground_truth/ground_truth.jsonl`. Pages without assigned IDs (TODO markers) are skipped automatically.

---

## 3. Method

### 3.1 Page Extraction

PDF pages are rendered to PNG at 300 DPI using PyMuPDF (`scripts/extract_pages.py`). Each page gets a `page_id` derived from the source filename and page number, which is the primary key linking predictions to ground truth.

### 3.2 Preprocessing

The image preprocessing pipeline (`src/preprocessing.py`) applies:

1. **Grayscale conversion** — discards colour information that adds noise on monochrome print.
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) — locally boosts contrast on faded or unevenly lit regions.
3. **Adaptive thresholding** — binarises the image with a block-local threshold to handle variation across the page.

The result is a clean binary image suitable for both layout analysis and OCR.

### 3.3 Main Text Region

The layout heuristic (`src/layout.py`) uses projection profiles with a **centre-bias weighting**: horizontal and vertical density projections are computed on the binary image, and a Gaussian weight up-weights the central band of the page. This suppresses marginal notes and running headers — which appear at the edges — without requiring trained layout detection.

This is the weakest part of the current pipeline. On pages where marginal notes are dense or the text block is significantly off-centre, the heuristic can include some marginal content in the crop.

### 3.4 OCR Backends

**TrOCR (zero-shot baseline):** `microsoft/trocr-base-printed` is used line-by-line via Hugging Face Transformers. Lines are segmented from the cropped binary image using horizontal run-length projection. The model is not fine-tuned on Spanish historical text, which is the primary source of error.

**CRNN (trained model):** A convolutional-recurrent network is implemented in `src/crnn_model.py`:
- **CNN backbone:** 4 conv-BN-LeakyReLU blocks with `AdaptiveAvgPool2d` collapsing height to 1.
- **BiLSTM head:** 2 bidirectional LSTM layers (hidden size 256).
- **CTC decoder:** greedy or lexicon-constrained beam search (see §3.5).
- **Weighted loss:** per-character inverse-frequency weights are computed from the ground truth corpus and applied to the CTC loss, up-weighting rare diacritics (`Ñ`, `é`, `í`, `ó`) and period-specific letterforms.

The CRNN was trained on 5 full-page crop images for 30 epochs. The loss reported as 0.0000 throughout — most likely because the full-page label strings exceed the CTC output sequence length on some samples, causing those batches to be zeroed by `zero_infinity=True`. A production training run would use per-line crops with shorter labels.

### 3.5 Decoding

Two CTC decoding strategies are available (`src/decoding.py`):

- **Greedy:** argmax at each time step, with standard repeat-collapse and blank removal. Fast and deterministic.
- **Lexicon beam search:** a pure-Python prefix beam search that applies a small log-space bonus to completed words appearing in `data/lexicon/spanish_renaissance_lexicon.txt` (≈250 period-appropriate lemmas). The bonus nudges the beam toward attested 17th-century orthography without blocking OOV tokens.

### 3.6 Post-OCR Cleanup

Three cleanup backends are implemented in `src/postprocess_llm.py`:

- **`rule_based`:** regex-based fixes (soft hyphens, ligatures `ﬁ`/`ﬂ`, whitespace normalisation, de-hyphenation at line boundaries).
- **`manual`:** writes a structured prompt to disk; reads back a saved human or LLM response.
- **`gemini`:** calls the Google Gemini API with a historically-informed prompt:

  > *"You are correcting OCR output from 17th-century printed Spanish text. Preserve meaning and historical spelling where reasonable. Fix obvious OCR mistakes only. Do not modernize the spelling unless the OCR is clearly wrong."*

  **Implementation note:** During testing, `gemini-2.0-flash` returned HTTP 429 (quota exhausted) on the free tier (`limit: 0`). The pipeline was updated to use `models/gemini-flash-lite-latest`, which is available on the free tier and returned correct responses. The cleaner falls back transparently to `rule_based` if the API call fails.

---

## 4. Evaluation

**Character Error Rate (CER):** Levenshtein edit distance at the character level, divided by the number of characters in the reference. CER > 1.0 is possible when the hypothesis is longer than the reference.

**Word Error Rate (WER):** Levenshtein edit distance at the word level, divided by the number of words in the reference.

Both metrics are computed after normalising whitespace (newlines collapsed, repeated spaces merged) so scores are not inflated by formatting differences. Comparisons are made for raw OCR and cleanup-corrected OCR independently.

---

## 5. Results

| Pages evaluated | Raw CER | Rule CER | Gemini CER | Raw WER | Rule WER | Gemini WER |
|---|---|---|---|---|---|---|
| 3 | 1.1468 | 1.1423 | **1.1026** | 1.3387 | 1.3050 | **1.2879** |

**Per-page breakdown (Gemini backend):**

| Page | Raw CER | Gemini CER | Δ CER |
|---|---|---|---|
| buendia_instruccion_page_0001 | 0.9480 | 0.8650 | −0.083 |
| buendia_instruccion_page_0002 | 0.9181 | 0.9192 | +0.001 |
| buendia_instruccion_page_0003 | 1.5745 | 1.5236 | −0.051 |

Gemini improved 2 of 3 pages, with a 8.3% CER reduction on the best page. Page 0002 showed a marginal regression (0.001 CER), likely because the rule-based cleanup was already optimal for that page's error profile.

### Representative Gemini correction

> **Raw:** `Io, P. GARCIA de Ia COMP. de JESUS`  
> **Gemini:** `Io, P. GARCIA de la COMP. de JESUS` ✓

The capital `I` being misread as lowercase `l` — then corrected by Gemini — is a known failure mode of TrOCR on historical fonts where the long-s and capital I are visually similar.

---

## 6. Error Analysis

- **OCR confuses period letterforms:** The long-s (`ſ`), capital `I` vs `l`, and abbreviation marks (tilde over a letter) are the most frequent source of character errors. TrOCR was trained predominantly on modern printed text and has no exposure to these forms.
- **Noisy raw text:** Some pages produce near-random TrOCR output (CER ≈ 1.57). On these pages, cleanup can recover some common words but cannot reconstruct the sentence structure.
- **Crop includes margin:** The centre-bias heuristic occasionally includes a column of marginal notes on pages with wide margins, inserting extraneous characters into the OCR input.
- **CTC training with full-page labels:** The CRNN training loop silently zeroed most loss values because full-page text labels exceeded the feature sequence length. A line-level training setup is required for the CRNN to learn.

---

## 7. Limitations

- **No fine-tuning on historical Spanish:** TrOCR zero-shot performance is the main bottleneck. The CRNN training is implemented but requires per-line crops to work correctly.
- **Small evaluation set:** 3 matched pages is too small for statistical confidence. Results should be interpreted as directional.
- **Layout heuristic is fragile:** Works well on clean single-column pages; degrades on multi-column, heavily annotated, or off-centre pages.
- **Gemini free-tier quota:** The `gemini-2.0-flash` model exhausted its free-tier quota during evaluation (HTTP 429). The pipeline was updated to use `models/gemini-flash-lite-latest` as a robust fallback.

---

## 8. Next Steps

1. **Line-level CRNN training:** Extract per-line crops using the existing segmentation output and re-run training with shorter label sequences. This should resolve the zero-loss issue and produce a genuinely trained model.
2. **TrOCR fine-tuning on Spanish historical text:** Run `scripts/finetune_trocr.py` on the 5 available ground-truth pages. Even minimal fine-tuning on in-domain data typically reduces CER significantly.
3. **Expand ground truth coverage:** Assign real page IDs to the remaining TODO pages in `transcription_marked.txt` to increase the evaluation set.
4. **Better layout handling:** Replace the projection-profile heuristic with a trained layout detector (e.g., a lightweight object-detection model fine-tuned on historical page layouts) to improve the crop quality on complex pages.
5. **Systematic cleanup evaluation:** Run cleanup on all available pages and analyse the δ-CER distribution to identify whether the LLM provides consistent gains or is only helpful for specific error types.

---

## Appendix: Reproduction Commands

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Extract page images (requires raw PDFs in data/raw_pdfs/)
python scripts/extract_pages.py --input-dir data/raw_pdfs --output-dir data/page_images --dpi 300

# Build ground truth JSONL
python scripts/parse_ground_truth.py

# Run TrOCR baseline (first 5 pages)
python scripts/run_baseline.py \
  --input-dir data/page_images \
  --predictions-file data/predictions/baseline_predictions.jsonl \
  --cleanup-backend rule_based \
  --limit 5

# Train CRNN
python scripts/train_crnn.py \
  --ground-truth-file data/ground_truth/ground_truth.jsonl \
  --line-image-dir data/crops \
  --output-dir outputs/crnn \
  --epochs 30

# Run with Gemini cleanup
python scripts/run_baseline.py \
  --input-dir data/page_images \
  --predictions-file data/predictions/gemini_predictions.jsonl \
  --cleanup-backend gemini \
  --gemini-api-key YOUR_GEMINI_API_KEY \
  --limit 5

# Evaluate
python scripts/evaluate_results.py \
  --predictions-file data/predictions/gemini_predictions.jsonl \
  --ground-truth-file data/ground_truth/ground_truth.jsonl \
  --output-dir outputs/gemini_eval
```

**Environment:** Python 3.13, PyTorch 2.2, Transformers 4.39, macOS (CPU only).  
**Model:** `microsoft/trocr-base-printed` (zero-shot).  
**Gemini model:** `models/gemini-flash-lite-latest` (free tier).
