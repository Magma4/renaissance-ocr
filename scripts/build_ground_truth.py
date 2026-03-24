from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import index_by, read_jsonl, sorted_paths, write_jsonl


PAGE_MARKER_RE = re.compile(r"^\s*===PAGE:\s*(?P<page_id>[a-zA-Z0-9_]+)\s*===\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/ground_truth/ground_truth.jsonl from page-level text files or a marked transcription."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of page-level .txt files named like buendia_instruccion_page_0001.txt",
    )
    source.add_argument(
        "--input-file",
        type=Path,
        help="Single transcription file. Use ===PAGE: page_id=== markers to define page breaks.",
    )
    source.add_argument(
        "--stdin",
        action="store_true",
        help="Read a marked transcription from stdin. Page breaks still need ===PAGE: page_id=== markers.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/ground_truth/ground_truth.jsonl"),
        help="Where to write the JSONL ground-truth file.",
    )
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        default=Path("data/page_images/manifest.jsonl"),
        help="Optional JSONL file with valid page_ids. Good choices are the extraction manifest or predictions file.",
    )
    parser.add_argument(
        "--no-reference-check",
        action="store_true",
        help="Skip validation against the reference JSONL file.",
    )
    parser.add_argument(
        "--skip-unresolved",
        action="store_true",
        help="Skip markers like TODO_buendia_instruccion_pdf_p2 instead of failing on them.",
    )
    return parser.parse_args()


def normalize_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def records_from_text_files(input_dir: Path) -> list[dict[str, str]]:
    text_files = sorted_paths(input_dir.glob("*.txt"))
    if not text_files:
        raise SystemExit(f"No .txt files found in {input_dir}")

    records: list[dict[str, str]] = []
    for path in text_files:
        page_id = path.stem
        text = normalize_page_text(path.read_text(encoding="utf-8"))
        if not text:
            continue
        records.append({"page_id": page_id, "text": text})
    return records


def is_unresolved_page_id(page_id: str) -> bool:
    return page_id.startswith("TODO_")


def records_from_marked_text(raw_text: str, skip_unresolved: bool = False) -> list[dict[str, str]]:
    current_page_id: str | None = None
    current_lines: list[str] = []
    records: list[dict[str, str]] = []

    def flush_current() -> None:
        nonlocal current_page_id, current_lines
        if current_page_id is None:
            return
        if skip_unresolved and is_unresolved_page_id(current_page_id):
            current_lines = []
            return
        page_text = normalize_page_text("\n".join(current_lines))
        if page_text:
            records.append({"page_id": current_page_id, "text": page_text})
        current_lines = []

    for line in raw_text.splitlines():
        if re.match(r"^\s*#", line):
            continue
        match = PAGE_MARKER_RE.match(line)
        if match:
            flush_current()
            current_page_id = match.group("page_id")
            continue
        if current_page_id is None and line.strip():
            raise SystemExit(
                "Found transcription text before the first page marker. "
                "Use lines like ===PAGE: buendia_instruccion_page_0001===."
            )
        current_lines.append(line)

    flush_current()
    if not records:
        raise SystemExit(
            "No page markers found. Use lines like ===PAGE: buendia_instruccion_page_0001===."
        )
    return records


def load_reference_page_ids(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(
            f"Reference JSONL not found: {path}. Use --no-reference-check to skip validation."
        )

    records = read_jsonl(path)
    page_ids = set()
    for index, record in enumerate(records, start=1):
        page_id = record.get("page_id")
        if not page_id:
            raise SystemExit(f"Reference record {index} in {path} is missing page_id.")
        page_ids.add(str(page_id))
    return page_ids


def validate_records(records: list[dict[str, str]], reference_page_ids: set[str] | None) -> None:
    index_by(records, "page_id")
    if reference_page_ids is None:
        return

    unknown_page_ids = [record["page_id"] for record in records if record["page_id"] not in reference_page_ids]
    if unknown_page_ids:
        preview = ", ".join(unknown_page_ids[:5])
        raise SystemExit(
            "Some page_ids do not match the reference file: "
            f"{preview}"
        )


def main() -> None:
    args = parse_args()

    if args.input_dir:
        records = records_from_text_files(args.input_dir)
    elif args.input_file:
        raw_text = args.input_file.read_text(encoding="utf-8")
        records = records_from_marked_text(raw_text, skip_unresolved=args.skip_unresolved)
    else:
        raw_text = sys.stdin.read()
        if not raw_text.strip():
            raise SystemExit("No text received on stdin.")
        records = records_from_marked_text(raw_text, skip_unresolved=args.skip_unresolved)

    reference_page_ids = None
    if not args.no_reference_check:
        reference_page_ids = load_reference_page_ids(args.reference_jsonl)

    validate_records(records, reference_page_ids)
    write_jsonl(args.output_file, records)

    print(f"Wrote {len(records)} records to {args.output_file}")
    if records:
        print(f"First page_id: {records[0]['page_id']}")


if __name__ == "__main__":
    main()
