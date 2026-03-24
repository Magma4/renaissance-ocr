from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from tqdm import tqdm

from src.crnn_model import CRNNOCRModel
from src.decoding import GreedyCTCDecoder, LexiconBeamSearchDecoder
from src.layout import crop_array, extract_main_text_region, segment_lines
from src.ocr_model import TrOCRPrintedBaseline
from src.postprocess_llm import run_cleanup
from src.preprocessing import grayscale_to_pil, prepare_line_for_ocr, preprocess_page
from src.utils import (
    CleanupConfig,
    CRNNConfig,
    OCRConfig,
    ensure_dir,
    index_by,
    list_image_files,
    read_jsonl,
    save_json,
    save_text,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the printed-text OCR baseline on extracted page images."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with page images from extract_pages.py.")
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="Optional manifest.jsonl from page extraction. If omitted, the script looks for one next to the images.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_pages"),
        help="Where to save binarized page images.",
    )
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=Path("data/crops"),
        help="Where to save the estimated main-text crops.",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        required=True,
        help="Output JSONL file for page-level OCR records.",
    )
    parser.add_argument(
        "--cleanup-backend",
        choices=["rule_based", "manual", "gemini"],
        default="rule_based",
        help="rule_based: fully local. manual: write prompts/read responses. gemini: call Gemini API.",
    )
    parser.add_argument(
        "--manual-prompt-dir",
        type=Path,
        default=None,
        help="Directory where manual/local cleanup prompts should be written.",
    )
    parser.add_argument(
        "--manual-response-dir",
        type=Path,
        default=None,
        help="Directory containing cleaned `page_id.txt` responses for the manual backend.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/trocr-base-printed",
        help="Hugging Face model name for TrOCR (ignored when --model-backend=crnn).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Line batch size for TrOCR inference.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N page images.")
    # CRNN options
    parser.add_argument(
        "--model-backend",
        choices=["trocr", "crnn"],
        default="trocr",
        help="OCR model backend: trocr (default) or crnn.",
    )
    parser.add_argument("--crnn-checkpoint", type=Path, default=None, help="Path to CRNN .pt checkpoint.")
    parser.add_argument("--crnn-vocab", type=Path, default=None, help="Path to CRNN vocab JSON.")
    parser.add_argument(
        "--crnn-decoder",
        choices=["greedy", "beam"],
        default="greedy",
        help="CTC decoding strategy for CRNN.",
    )
    parser.add_argument(
        "--lexicon-path",
        type=Path,
        default=Path("data/lexicon/spanish_renaissance_lexicon.txt"),
        help="Wordlist for lexicon beam search (optional).",
    )
    # Gemini options
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Google Gemini API key (can also be set via GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--debug-page-id",
        type=str,
        default=None,
        help="Save intermediate OCR artifacts for one page id.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("outputs/debug"),
        help="Root directory for page-level debug artifacts.",
    )
    return parser.parse_args()


def load_manifest_index(manifest_file: Path | None) -> dict[str, dict]:
    if manifest_file is None or not manifest_file.exists():
        return {}
    return index_by(read_jsonl(manifest_file), "page_id")


def save_debug_artifacts(
    debug_root: Path,
    page_id: str,
    gray_crop,
    line_arrays,
    line_images,
    line_predictions,
    raw_ocr: str,
    cleaned_ocr: str,
    crop_bbox: list[int],
    crop_reason: str,
) -> None:
    page_debug_dir = ensure_dir(debug_root / page_id)
    grayscale_to_pil(gray_crop).save(page_debug_dir / "crop.png")

    line_rows = []
    for index, (segment, ocr_image, prediction) in enumerate(
        zip(line_arrays, line_images, line_predictions),
        start=1,
    ):
        segment_name = f"line_{index:04d}_segment.png"
        ocr_name = f"line_{index:04d}_ocr.png"
        grayscale_to_pil(segment).save(page_debug_dir / segment_name)
        ocr_image.save(page_debug_dir / ocr_name)
        line_rows.append(
            {
                "line_index": index,
                "segment_image": segment_name,
                "ocr_image": ocr_name,
                "text": prediction,
            }
        )

    write_jsonl(page_debug_dir / "line_ocr.jsonl", line_rows)
    save_text(page_debug_dir / "raw_ocr.txt", raw_ocr + "\n")
    save_text(page_debug_dir / "cleaned_ocr.txt", cleaned_ocr + "\n")
    save_json(
        page_debug_dir / "debug.json",
        {
            "page_id": page_id,
            "crop_bbox": crop_bbox,
            "crop_reason": crop_reason,
            "line_count": len(line_images),
        },
    )


def main() -> None:
    args = parse_args()
    image_paths = list_image_files(args.input_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No page images found in {args.input_dir}")

    manifest_file = args.manifest_file or (args.input_dir / "manifest.jsonl")
    manifest_index = load_manifest_index(manifest_file)

    processed_dir = ensure_dir(args.processed_dir)
    crop_dir = ensure_dir(args.crop_dir)
    cleanup_config = CleanupConfig(
        backend=args.cleanup_backend,
        manual_prompt_dir=args.manual_prompt_dir,
        manual_response_dir=args.manual_response_dir,
        gemini_api_key=args.gemini_api_key,
    )

    # ── Load OCR model ────────────────────────────────────────────────
    if args.model_backend == "crnn":
        crnn_config = CRNNConfig(
            checkpoint_path=str(args.crnn_checkpoint) if args.crnn_checkpoint else None,
            vocab_path=str(args.crnn_vocab) if args.crnn_vocab else None,
            decoder=args.crnn_decoder,
            lexicon_path=str(args.lexicon_path) if args.lexicon_path and args.lexicon_path.exists() else None,
        )
        model = CRNNOCRModel(crnn_config)  # type: ignore[assignment]
        if crnn_config.checkpoint_path is None or crnn_config.vocab_path is None:
            raise SystemExit(
                "CRNN backend requires --crnn-checkpoint and --crnn-vocab. "
                "Run scripts/train_crnn.py first."
            )
        print(f"Using CRNN backend | decoder={args.crnn_decoder}")
    else:
        try:
            model = TrOCRPrintedBaseline(  # type: ignore[assignment]
                OCRConfig(
                    model_name=args.model_name,
                    batch_size=args.batch_size,
                )
            )
        except OSError as exc:
            raise SystemExit(
                "Could not load the TrOCR model. If this is the first run, make sure the Hugging Face download can complete."
            ) from exc
        print(f"Using TrOCR backend | model={args.model_name}")

    records: list[dict] = []
    for image_path in tqdm(image_paths, desc="OCR"):
        page_id = image_path.stem
        manifest_row = manifest_index.get(page_id, {})

        with Image.open(image_path) as image:
            original_image = image.convert("RGB")
            views = preprocess_page(original_image)

        processed_binary_path = processed_dir / f"{page_id}_binary.png"
        grayscale_to_pil(views["binary"]).save(processed_binary_path)

        crop = extract_main_text_region(views["gray"], views["binary"])
        gray_crop = crop_array(views["gray"], crop)
        binary_crop = crop_array(views["binary"], crop)
        grayscale_to_pil(gray_crop).save(crop_dir / f"{page_id}_crop.png")

        line_arrays = segment_lines(gray_crop, binary_crop)
        line_images = [prepare_line_for_ocr(line) for line in line_arrays]
        line_predictions = model.predict_lines(line_images)
        raw_ocr = "\n".join(line for line in line_predictions if line)
        cleanup = run_cleanup(page_id=page_id, raw_text=raw_ocr, config=cleanup_config)

        if args.debug_page_id and page_id == args.debug_page_id:
            debug_page_dir = ensure_dir(args.debug_dir / page_id)
            original_image.save(debug_page_dir / "original.png")
            grayscale_to_pil(views["binary"]).save(debug_page_dir / "processed_binary.png")
            save_debug_artifacts(
                debug_root=args.debug_dir,
                page_id=page_id,
                gray_crop=gray_crop,
                line_arrays=line_arrays,
                line_images=line_images,
                line_predictions=line_predictions,
                raw_ocr=raw_ocr,
                cleaned_ocr=str(cleanup["cleaned_text"] or ""),
                crop_bbox=list(crop.bbox),
                crop_reason=crop.reason,
            )

        records.append(
            {
                "page_id": page_id,
                "image_path": str(image_path),
                "source_pdf": manifest_row.get("source_pdf"),
                "page_number": manifest_row.get("page_number"),
                "crop_bbox": list(crop.bbox),
                "crop_reason": crop.reason,
                "line_count": len(line_images),
                "cleanup_backend_requested": cleanup["backend_requested"],
                "cleanup_backend_used": cleanup["backend_used"],
                "cleanup_prompt_path": cleanup["prompt_path"],
                "raw_ocr": raw_ocr,
                "cleaned_ocr": cleanup["cleaned_text"],
            }
        )

    write_jsonl(args.predictions_file, records)
    print(f"Wrote {len(records)} page records to {args.predictions_file}")


if __name__ == "__main__":
    main()
