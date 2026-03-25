# renAIssance: Printed OCR

**Organization:** HumanAI Foundation  
**Applicant:** Raymond Frimpong Amoateng  
**Project:** RenAIssance: Improving OCR for Early Modern Printed Spanish Sources

---

## About Me

I am a Computer Science student with a strong interest in machine learning applied to cultural heritage problems. I have been working with computer vision and NLP for a couple of years, not just as coursework but because I find the problem of making historical documents machine-readable genuinely interesting. Most OCR research focuses on modern text, so the failure modes on 17th-century typefaces are actually a different and underexplored problem.

I am comfortable working in Python, PyTorch, and Transformers. I have built sequence-to-sequence models before and I know the CTC loss formulation well enough to have debugged it from scratch. I can work independently and I know when to ask questions.

---

## What Drew Me to This Project

I have been thinking about why modern OCR models fail so badly on old printed text. It is not just that the scans are degraded. The letterforms themselves are different. The long-s looks like an f. Capital I and lowercase l are almost identical. Abbreviation marks over letters do not appear in any modern training corpus. A model like TrOCR sees these and just guesses based on its prior over modern text.

For the test submission, I ran microsoft/trocr-base-printed zero-shot on three pages from the Buendia Instruccion source and got a Character Error Rate of about 1.15. That is not a calibration issue. The model is producing things like "WWW.LLM" and "GARDORIZIO ON 6020" where the ground truth is actual 17th-century Spanish prose. The main bottleneck is domain mismatch, not architecture.

The problem I want to work on this summer is specifically: how do you bring OCR performance to a usable level for these sources with limited labelled data?

---

## What I Have Already Built

For the evaluation test, I put together a full pipeline before submitting this proposal.

A CRNN model with a 4-block CNN backbone, AdaptiveAvgPool, BiLSTM, and CTC, implemented from scratch in PyTorch, with inverse-frequency character weighting so rare diacritics and ligatures do not get swamped during training.

A lexicon-constrained beam search decoder written in pure Python without any C++ dependencies. It applies a small log-space bonus to completed words at word boundaries using a curated period Spanish wordlist of about 250 lemmas. It degrades gracefully to greedy decoding on out-of-vocabulary tokens.

Gemini API integration for post-OCR cleanup, with a prompt specifically tuned to preserve period spelling while correcting obvious recognition errors. I ran into a gemini-2.0-flash free-tier quota limit during testing and switched to models/gemini-flash-lite-latest, which worked. The fallback logic I built actually got used in practice, which was a useful real-world test of the robustness.

Fine-tuning scripts for TrOCR using the Hugging Face Seq2SeqTrainer.

A CER/WER evaluation framework comparing raw, rule-based, and Gemini cleanup results.

Results on 3 matched evaluation pages:

| Backend | CER | WER |
|---|---|---|
| TrOCR zero-shot raw | 1.1468 | 1.3387 |
| Rule-based cleanup | 1.1423 | 1.3050 |
| Gemini cleanup | 1.1026 | 1.2879 |

Gemini correctly caught things like capital I being misread as lowercase l ("Ia" to "la"), which is a real 17th-century OCR failure mode.

The repo is at: https://github.com/Magma4/renaissance-ocr

---

## What I Plan to Build During GSoC

The main gap right now is that CRNN training does not work properly at the page level. The ground truth text is too long relative to the CTC output sequence, so most batches get zeroed by zero_infinity=True. The model does not learn. Fixing this requires per-line crops with per-line text labels, which means better line segmentation.

**Phase 1: Better data and line-level training (Weeks 1 to 4)**

The current line segmentation uses horizontal projection profiles. It works on clean single-column pages but breaks on dense or multi-column pages. I want to replace it with a lightweight trainable approach, probably a U-Net or YOLO-based detector fine-tuned on a small set of annotated rows. Once I have clean line crops, I can train the CRNN with labels that fit within the CTC sequence length and actually see meaningful loss curves.

I will also work on expanding ground truth coverage. Right now there are 19 TODO markers in transcription_marked.txt for pages without assigned IDs. Getting even half of those transcribed would meaningfully increase the training set.

**Phase 2: Model improvements (Weeks 5 to 8)**

Once training works at the line level, I want to try a few things. Replacing the BiLSTM head with a lightweight cross-attention layer, similar in spirit to a miniature TrOCR but staying in the CRNN framework. Experimenting with self-supervised pre-training on unlabelled page crops before fine-tuning on ground truth, since there are many pages without transcriptions that can still be used to learn visual features. Evaluating whether augmentation like synthetic degradation, rotation, and noise helps generalization across sources.

**Phase 3: System integration and evaluation (Weeks 9 to 12)**

Full pipeline evaluation across all sources, not just Buendia. A proper comparison between fine-tuned TrOCR and the trained CRNN, with statistical confidence intervals. Documentation and a cleaned-up public-facing demo so other digital humanities researchers can actually use the pipeline.

---

## Timeline

| Week | Work |
|---|---|
| 1 | Set up communication with mentors, finalize ground truth expansion plan, annotate 5 to 10 pages for line detection |
| 2 to 3 | Train lightweight line detector, replace projection heuristic with model-based segmentation |
| 4 | End-to-end CRNN training at line level, verify loss curves look healthy |
| 5 to 6 | Experiment with augmentation, try cross-attention CRNN variant |
| 7 | Midterm evaluation, document results so far |
| 8 | Self-supervised pre-training experiments |
| 9 to 10 | Full pipeline evaluation across all 6 sources |
| 11 | Fine-tuned TrOCR vs CRNN comparison, statistical analysis |
| 12 | Final documentation, demo, cleanup, final evaluation |

---

## Practical Notes

This runs as a standard 175-hour GSoC project. I have a MacBook with an M-series chip for development. Compute-heavy training I will run on a cloud VM where I have GCP credits. I am available about 15 hours per week during the 12-week GSoC period to complete the 175 hours. I prefer async communication but am happy to do weekly video syncs if that works for the mentors. I have already read most of the source material in the data directory and have a rough sense of which sources will be hardest. The Porcones legal documents are much denser than the Buendia pages.

---

## Why This Project Matters

There are thousands of 17th-century Spanish documents in archives that are essentially inaccessible to computational analysis because the OCR quality is too low to be useful. A pipeline that can get CER below 0.3 on clean pages, which I think is achievable with proper fine-tuning, would make those documents searchable, linkable, and analyzable in ways that are not currently possible. That is what I want to actually ship this summer.
