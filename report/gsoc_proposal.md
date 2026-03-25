# renAIssance: Printed OCR

**Organization:** HumanAI Foundation
**Applicant:** Raymond Frimpong Amoateng
**Email:** raymondamoateng@gmail.com
**GitHub:** https://github.com/Magma4
**Project:** RenAIssance: Improving OCR for Early Modern Printed Spanish Sources

---

## Synopsis

This project aims to build an end-to-end OCR pipeline designed specifically for 17th-century printed Spanish sources. Traditional OCR engines and modern vision-language models struggle with historical typefaces, such as the long-s, archaic ligatures, and abbreviation marks. This proposal details a custom CNN-RNN model trained with inverse-frequency weighted CTC loss to handle rare diacritics, paired with a lexicon-constrained beam search decoder and Gemini LLM post-correction. The goal is to reduce the Character Error Rate to a usable level for digital humanities researchers on limited labelled data.

---

## Benefits to Community

There are thousands of 17th-century Spanish documents in archives that are essentially inaccessible to computational analysis because the OCR quality is too low to be useful. A pipeline that can get CER below 0.3 on clean pages, which I think is achievable with proper fine-tuning, would make those documents searchable, linkable, and analyzable in ways that are not currently possible. This project will deliver a fully open-source pipeline that Google, HumanAI, and society at large can use to unlock historical Spanish texts.

---

## Deliverables

The main gap right now is that CRNN training does not work properly at the page level. Fixing this requires per-line crops with per-line text labels, which means better line segmentation.

**Phase 1: Better data and line-level training (Weeks 1 to 4)**
*   Train a lightweight trainable line detector (U-Net or YOLO) fine-tuned on annotated rows. **[Required]**
*   Expand ground truth coverage by transcribing the remaining 19 unassigned pages. **[Required]**

**Phase 2: Model improvements (Weeks 5 to 8)**
*   End-to-end line-level CRNN training with the new data. **[Required]**
*   Experiment with self-supervised pre-training on unlabelled page crops. **[Optional]**
*   Replace BiLSTM head with a lightweight cross-attention layer. **[Optional]** 

**Phase 3: System integration and evaluation (Weeks 9 to 12)**
*   Full pipeline evaluation across all sources. **[Required]**
*   Clean public-facing demo and documentation. **[Required]**

### Timeline

| Week    | Work                                                                                                               | Status |
| ------- | ------------------------------------------------------------------------------------------------------------------ | ------ |
| 1       | Set up communication with mentors, finalize ground truth expansion plan, annotate 5 to 10 pages for line detection | Required |
| 2 to 3  | Train lightweight line detector, replace projection heuristic with model-based segmentation                        | Required |
| 4       | End-to-end CRNN training at line level, verify loss curves look healthy                                            | Required |
| 5 to 6  | Experiment with augmentation, try cross-attention CRNN variant                                                     | Optional |
| 7       | Midterm evaluation, document results so far                                                                        | Required |
| 8       | Self-supervised pre-training experiments                                                                           | Optional |
| 9 to 10 | Full pipeline evaluation across all 6 sources                                                                      | Required |
| 11      | Fine-tuned TrOCR vs CRNN comparison, statistical analysis                                                          | Required |
| 12      | Final documentation, demo, cleanup, final evaluation                                                               | Required |

---

## Related Work

Existing OCR engines like Tesseract and Kraken are widely used in digital humanities but often require extensive fine-tuning and layout annotation to work well on historical texts. Modern vision-transformer approaches like TrOCR provide a strong zero-shot baseline but still suffer from domain mismatch on 17th-century letterforms. This project builds upon these foundations by combining a lightweight, domain-specific CRNN architecture with modern LLM post-processing (Gemini), demonstrating that a hybrid approach—specialized recognition and contextual correction—can significantly outperform zero-shot monolithic models. 

For the evaluation test, I put together a full preliminary pipeline before submitting this proposal. Results on 3 matched evaluation pages showed Gemini LLM cleanup reduced CER from 1.1468 (TrOCR zero-shot) to 1.1026. The pipeline code is at: https://github.com/Magma4/renaissance-ocr

---

## Biographical Information

I am a Computer Science student with a strong interest in machine learning applied to cultural heritage problems. I have been working with computer vision and NLP for a couple of years, not just as coursework but because I find the problem of making historical documents machine-readable genuinely interesting. Most OCR research focuses on modern text, so the failure modes on 17th-century typefaces are actually a different and underexplored problem.

I am comfortable working in Python, PyTorch, and Transformers. I have built sequence-to-sequence models before and I know the CTC loss formulation well enough to have debugged it from scratch. I can work independently and I know when to ask questions.

---

## Practical Notes

This runs as a standard 175-hour GSoC project. I have a MacBook with an M-series chip for development. Compute-heavy training I will run on a cloud VM where I have GCP credits. I am available about 15 hours per week during the 12-week GSoC period to complete the 175 hours. I prefer async communication but am happy to do weekly video syncs if that works for the mentors.
