import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, read_jsonl, write_jsonl
from src.vlm_pipeline import GeminiVLMExtractor, GEMINI_AVAILABLE

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Gemini VLM pipeline over extracted manuscript images."
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "page_images" / "manifest.jsonl", help="Path to manifest JSONL.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "predictions" / "vlm_results.jsonl", help="Output JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images for a quick smoke test.")
    return parser.parse_args()

def main():
    if not GEMINI_AVAILABLE:
        raise SystemExit("Error: google-genai is not installed.")

    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    ensure_dir(args.output.parent)
    manifest = read_jsonl(args.manifest)
    
    if args.limit:
        manifest = manifest[:args.limit]

    print(f"Loaded {len(manifest)} pages from {args.manifest}")
    
    extractor = GeminiVLMExtractor()
    results = []

    for item in tqdm(manifest, desc="Extracting text via VLM"):
        image_path = Path(item["image_path"])
        if not image_path.exists():
            print(f"Warning: Image missing {image_path}")
            continue

        try:
            transcription = extractor.extract_text(image_path)
            results.append({
                "page_id": item["page_id"],
                "source_pdf": item["source_pdf"],
                "vlm_text": transcription
            })
        except Exception as e:
            print(f"Error on {item['page_id']}: {e}")

    write_jsonl(args.output, results)
    print(f"Saved {len(results)} transcriptions to {args.output}")

if __name__ == "__main__":
    main()
