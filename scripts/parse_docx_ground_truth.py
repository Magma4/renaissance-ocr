import argparse
import re
import sys
from pathlib import Path

try:
    import docx
except ImportError:
    print("Please install python-docx: pip install python-docx")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, write_jsonl, slugify_name, make_page_id

PAGE_HEADER_RE = re.compile(r"^PDF p(\d+)", flags=re.IGNORECASE)

def parse_docx(docx_path: Path):
    doc = docx.Document(docx_path)
    source_name = re.sub(r"[_ ]*[tT]ranscription$", "", docx_path.stem)
    
    records = []
    current_page_id = None
    current_lines = []

    def flush():
        if current_page_id is not None:
            page_text = "\n".join(current_lines).strip()
            if page_text:
                records.append({"page_id": current_page_id, "text": page_text})

    for p in doc.paragraphs:
        line = p.text.strip()
        if not line:
            continue
            
        m = PAGE_HEADER_RE.match(line)
        if m:
            flush()
            page_number = int(m.group(1))
            current_page_id = make_page_id(source_name, page_number)
            current_lines = []
            continue
            
        if current_page_id is None:
            continue
            
        current_lines.append(line)
        
    flush()
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(ROOT / "data/ground_truth/raw_docx"))
    parser.add_argument("--output-file", type=Path, default=Path(ROOT / "data/ground_truth/ground_truth.jsonl"))
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        raise SystemExit(f"Directory not found: {args.input_dir}")
        
    ensure_dir(args.output_file.parent)
    
    all_records = []
    # Only parsing handwritten ones, we'll exclude the print ones if they are mixed
    # We'll just parse whatever is in the directory
    for docx_path in sorted(args.input_dir.glob("*.docx")):
        if docx_path.name.startswith("~"):
            continue
        records = parse_docx(docx_path)
        all_records.extend(records)
        print(f"Parsed {len(records)} pages from {docx_path.name}")
        
    write_jsonl(args.output_file, all_records)
    print(f"Total: Wrote {len(all_records)} ground-truth records to {args.output_file}")

if __name__ == "__main__":
    main()
