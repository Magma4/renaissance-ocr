# Ground truth notes

The evaluation script expects one JSONL record per page in `ground_truth.jsonl`:

```json
{"page_id":"buendia_instruccion_page_0001","text":"page transcription here"}
{"page_id":"buendia_instruccion_page_0002","text":"next page transcription here"}
```

`page_id` has to match the extracted page image or prediction record exactly.

There are three practical ways to build this file:

1. If you already have one text file per page, name them with the page id and run:

```bash
python scripts/build_ground_truth.py \
  --input-dir data/ground_truth/page_texts
```

2. If you have one longer transcription, add page markers like this:

```text
===PAGE: buendia_instruccion_page_0001===
primer texto...

===PAGE: buendia_instruccion_page_0002===
segundo texto...
```

Then run:

```bash
python scripts/build_ground_truth.py \
  --input-file data/ground_truth/transcription_marked.txt
```

You can also paste marked text straight into the script:

```bash
python scripts/build_ground_truth.py --stdin
```

3. If your source transcriptions are in `.docx` files under `Print/`, prepare them like this:

```bash
python scripts/prepare_docx_ground_truth.py
```

That script will:

- copy the `.docx` files into `data/ground_truth/raw_docx/`
- extract plain text into `data/ground_truth/extracted_txt/`
- write `data/ground_truth/transcription_marked.txt`
- write `data/ground_truth/page_id_reference.txt`

The marked file uses temporary markers like:

```text
===PAGE: TODO_buendia_instruccion_pdf_p2===
```

Replace those `TODO_...` markers with real page ids from `page_id_reference.txt`, then build the final JSONL:

```bash
python scripts/build_ground_truth.py \
  --input-file data/ground_truth/transcription_marked.txt \
  --reference-jsonl data/predictions/baseline_predictions.jsonl \
  --skip-unresolved
```

`--skip-unresolved` is helpful while you are still mapping pages by hand. It will only write the pages you have already resolved.

By default the builder checks page ids against `data/page_images/manifest.jsonl`, which helps catch typos before evaluation.
