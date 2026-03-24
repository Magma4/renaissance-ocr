from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz
from tqdm import tqdm

from src.utils import ensure_dir, make_page_id, slugify_name, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PDF scans to page images and write a small manifest next to them."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing PDF files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for extracted PNG page images.")
    parser.add_argument("--dpi", type=int, default=300, help="Render resolution.")
    parser.add_argument("--max-pages-per-pdf", type=int, default=None, help="Useful for quick smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    pdf_paths = sorted(args.input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {args.input_dir}")

    manifest: list[dict[str, str | int]] = []
    for pdf_path in tqdm(pdf_paths, desc="PDFs"):
        source_stem = slugify_name(pdf_path.stem)
        with fitz.open(pdf_path) as document:
            page_total = len(document)
            if args.max_pages_per_pdf is None:
                page_limit = page_total
            else:
                page_limit = min(page_total, args.max_pages_per_pdf)

            matrix = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)
            for page_index in range(page_limit):
                page_number = page_index + 1
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                page_id = make_page_id(source_stem, page_number)
                output_path = output_dir / f"{page_id}.png"
                pixmap.save(output_path)
                manifest.append(
                    {
                        "page_id": page_id,
                        "source_pdf": str(pdf_path),
                        "page_number": page_number,
                        "image_path": str(output_path),
                    }
                )

    write_jsonl(output_dir / "manifest.jsonl", manifest)
    print(f"Extracted {len(manifest)} pages into {output_dir}")


if __name__ == "__main__":
    main()
