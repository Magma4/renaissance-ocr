# RenAIssance Test I report notes

## Title

Printed OCR baseline for early modern Spanish sources

## 1. Problem and scope

Start by saying exactly what this project is trying to do and what it is not trying to do.

- printed historical OCR only
- main text only
- marginalia ignored as much as possible
- LLM used only after OCR, not as the recognizer

One short paragraph here is enough. This section should make the scope feel controlled, not ambitious for its own sake.

## 2. Data

Write down which PDFs you used, how many pages you actually evaluated, and what kind of pages they were. It helps to mention if the scans are fairly clean, badly skewed, full of side notes, and so on.

Also explain how ground truth was prepared. If only part of a source was transcribed, say that clearly.

## 3. Method

### 3.1 Page extraction

Note the PDF-to-image step, DPI, and any practical choices that mattered.

### 3.2 Preprocessing

Describe the image cleanup briefly. Keep it concrete:

- grayscale
- contrast adjustment
- denoising
- adaptive thresholding

### 3.3 Main text region

Explain the crop heuristic in plain language. The important part is that the system tries to stay on the central printed block instead of chasing marginal notes.

Be honest if this is the weakest part of the current baseline.

### 3.4 OCR model

State the model and why you picked it. If it is still zero-shot, say so directly. That is fine for a first pass.

### 3.5 Cleanup

Describe the rule-based cleanup first, then the optional prompt-based cleanup. Make it explicit that this stage only edits OCR output after recognition.

## 4. Evaluation

Define CER and WER in one or two lines. Then explain what was compared:

- raw OCR vs ground truth
- cleaned OCR vs ground truth

If you normalized whitespace for scoring, mention that too.

## 5. Results

Put the main table here. Keep it small.

Suggested columns:

- evaluated pages
- raw CER
- cleaned CER
- raw WER
- cleaned WER

After the table, include two or three examples that show what improved and what did not.

## 6. Error analysis

This section matters more than inflated claims. A few grounded notes are better than a long generic list.

Useful categories:

- crop included some marginal text
- line segmentation broke on dense pages
- OCR confused similar letterforms
- degraded scans hurt both recognition and cleanup
- cleanup occasionally left bad text alone because the rules were too cautious

## 7. Limitations

Keep this short and specific. A few examples:

- no fine-tuning yet
- layout handling is still heuristic
- evaluation set is small
- some pages are not well represented by a single main-text crop

## 8. Next steps

Write the next steps in the same practical tone as the repo:

- improve crop quality on pages with side notes
- build a slightly better evaluation subset
- fine-tune on line crops if time allows
- test whether cleanup helps consistently or only on a few pages

## Appendix

Include the commands you ran, model name, environment notes, and anything else needed to reproduce the baseline without turning the main report into a log file.
