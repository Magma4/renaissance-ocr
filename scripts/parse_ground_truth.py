"""Parse transcription_marked.txt into the ground_truth.jsonl format.

The marked transcription file uses the format:
    ===PAGE: <page_id>===
    <page text...>
    ===PAGE: <next_page_id>===
    ...

Comment lines starting with # and END OF EXTRACT markers are stripped.
TODO_ markers are left intact but warn – they represent pages not yet assigned
a real page_id.

Usage
-----
    python scripts/parse_ground_truth.py \\
        --input-file data/ground_truth/transcription_marked.txt \\
        --output-file data/ground_truth/ground_truth.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import write_jsonl


PAGE_HEADER_RE = re.compile(r"^===PAGE:\s*(.+?)\s*===\s*$")
SKIP_LINE_RE = re.compile(r"^(#|END OF EXTRACT)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert transcription_marked.txt to ground_truth.jsonl."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/ground_truth/transcription_marked.txt"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/ground_truth/ground_truth.jsonl"),
    )
    parser.add_argument(
        "--skip-todo",
        action="store_true",
        default=True,
        help="Skip pages whose page_id still starts with TODO_ (default: True).",
    )
    return parser.parse_args()


def parse_transcription(text: str, skip_todo: bool = True) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current_page_id: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_page_id is None:
            return
        page_text = "\n".join(current_lines).strip()
        if not page_text:
            return
        if skip_todo and current_page_id.startswith("TODO_"):
            print(f"  [skip] {current_page_id} — no page_id assigned yet", file=sys.stderr)
            return
        records.append({"page_id": current_page_id, "text": page_text})

    for line in text.splitlines():
        m = PAGE_HEADER_RE.match(line)
        if m:
            flush()
            current_page_id = m.group(1).strip()
            current_lines = []
            continue

        if SKIP_LINE_RE.match(line.strip()):
            continue

        if current_page_id is not None:
            current_lines.append(line.rstrip())

    flush()
    return records


def main() -> None:
    args = parse_args()
    if not args.input_file.exists():
        raise SystemExit(f"Input file not found: {args.input_file}")

    text = args.input_file.read_text(encoding="utf-8")
    records = parse_transcription(text, skip_todo=args.skip_todo)
    write_jsonl(args.output_file, records)
    print(f"Wrote {len(records)} ground-truth records to {args.output_file}")


if __name__ == "__main__":
    main()
