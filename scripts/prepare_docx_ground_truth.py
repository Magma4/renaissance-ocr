from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docx import Document

from src.utils import ensure_dir, read_jsonl, save_text, slugify_name, sorted_paths


DOCX_PAGE_MARKER_RE = re.compile(r"^\s*PDF p\d+(?:\s*[–-]\s*(?:left|right))?\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy printed transcription .docx files into data/ground_truth and turn them into editable marked text."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Print"),
        help="Folder with the original printed transcription .docx files.",
    )
    parser.add_argument(
        "--raw-docx-dir",
        type=Path,
        default=Path("data/ground_truth/raw_docx"),
        help="Where to copy the source .docx files.",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path("data/ground_truth/extracted_txt"),
        help="Where to save plain-text extracts from the .docx files.",
    )
    parser.add_argument(
        "--marked-file",
        type=Path,
        default=Path("data/ground_truth/transcription_marked.txt"),
        help="Combined editable transcription file with TODO page markers.",
    )
    parser.add_argument(
        "--page-id-reference",
        type=Path,
        default=Path("data/ground_truth/page_id_reference.txt"),
        help="A plain-text list of available page ids to help with manual mapping.",
    )
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        default=Path("data/predictions/baseline_predictions.jsonl"),
        help="JSONL file that defines the page_ids you want ground truth to match.",
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=Path("data/page_images/manifest.jsonl"),
        help="Fallback page-id source if the main reference JSONL is missing.",
    )
    return parser.parse_args()


def docx_to_text(docx_path: Path) -> str:
    document = Document(docx_path)
    paragraphs: list[str] = []
    for paragraph in document.paragraphs:
        paragraphs.append(paragraph.text)
    return "\n".join(paragraphs).replace("\r\n", "\n").replace("\r", "\n")


def clean_source_name(name: str) -> str:
    cleaned = re.sub(r"\s+transcription$", "", name, flags=re.IGNORECASE)
    return cleaned.strip()


def split_docx_text(raw_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_marker: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_marker, current_lines
        if current_marker is None:
            current_lines = []
            return
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_marker, text))
        current_lines = []

    for line in raw_text.splitlines():
        if DOCX_PAGE_MARKER_RE.match(line):
            flush()
            current_marker = line.strip()
            continue
        if current_marker is None:
            continue
        current_lines.append(line)

    flush()
    return sections


def marker_to_todo_page_id(source_key: str, marker: str) -> str:
    marker_key = slugify_name(marker.replace("PDF ", "pdf_"))
    return f"TODO_{source_key}_{marker_key}"


def load_reference_page_ids(primary_path: Path, fallback_path: Path) -> dict[str, list[str]]:
    source_path = primary_path if primary_path.exists() else fallback_path
    if not source_path.exists():
        raise SystemExit(
            f"Could not find a reference JSONL at {primary_path} or {fallback_path}."
        )

    grouped: dict[str, list[str]] = defaultdict(list)
    for record in read_jsonl(source_path):
        page_id = str(record.get("page_id", "")).strip()
        if not page_id:
            continue
        source_key = page_id.rsplit("_page_", 1)[0]
        grouped[source_key].append(page_id)

    return dict(sorted(grouped.items()))


def build_marked_text(documents: list[dict], page_ids_by_source: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for index, document in enumerate(documents):
        if index:
            lines.append("")
            lines.append("")

        source_key = document["source_key"]
        available_page_ids = page_ids_by_source.get(source_key, [])
        if available_page_ids:
            lines.append(f"# {source_key}")
            lines.append("# Available page_ids for this source:")
            for page_id in available_page_ids:
                lines.append(f"# - {page_id}")
            lines.append("")

        for marker, text in document["sections"]:
            lines.append(f"===PAGE: {marker_to_todo_page_id(source_key, marker)}===")
            lines.append(text.strip())
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_reference_note(page_ids_by_source: dict[str, list[str]]) -> str:
    lines = [
        "Available page_ids",
        "",
        "Use these exact ids when you replace the TODO markers in transcription_marked.txt.",
        "",
    ]
    for source_key, page_ids in page_ids_by_source.items():
        lines.append(f"{source_key}:")
        for page_id in page_ids:
            lines.append(f"  - {page_id}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    raw_docx_dir = ensure_dir(args.raw_docx_dir)
    text_dir = ensure_dir(args.text_dir)

    docx_paths = sorted_paths(args.input_dir.glob("*.docx"))
    if not docx_paths:
        raise SystemExit(f"No .docx files found in {args.input_dir}")

    page_ids_by_source = load_reference_page_ids(args.reference_jsonl, args.manifest_jsonl)

    documents: list[dict] = []
    for docx_path in docx_paths:
        copied_path = raw_docx_dir / docx_path.name
        shutil.copy2(docx_path, copied_path)

        source_name = clean_source_name(docx_path.stem)
        source_key = slugify_name(source_name)
        raw_text = docx_to_text(docx_path)
        save_text(text_dir / f"{source_key}.txt", raw_text)

        sections = split_docx_text(raw_text)
        documents.append(
            {
                "source_name": source_name,
                "source_key": source_key,
                "docx_path": str(copied_path),
                "sections": sections,
            }
        )

    save_text(args.marked_file, build_marked_text(documents, page_ids_by_source))
    save_text(args.page_id_reference, build_reference_note(page_ids_by_source))

    print(f"Copied {len(documents)} .docx files to {raw_docx_dir}")
    print(f"Wrote editable transcription file to {args.marked_file}")
    print(f"Wrote page-id reference note to {args.page_id_reference}")


if __name__ == "__main__":
    main()
