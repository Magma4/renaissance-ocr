from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_records, load_ground_truth_records, save_evaluation_artifacts
from src.utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate page-level OCR predictions against a ground-truth JSONL file."
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        required=True,
        help="JSONL file written by scripts/run_baseline.py.",
    )
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        required=True,
        help="JSONL file with one record per page: {'page_id': ..., 'text': ...}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write summary.json, comparisons.jsonl, and sample examples.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=5,
        help="How many side-by-side examples to save for quick inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_records = read_jsonl(args.predictions_file)
    ground_truth_records = load_ground_truth_records(args.ground_truth_file)
    evaluation = evaluate_records(prediction_records, ground_truth_records)
    save_evaluation_artifacts(evaluation, args.output_dir, example_limit=args.example_limit)

    summary = evaluation["summary"]
    print(f"Evaluated {summary['evaluated_pages']} pages")
    print(f"Average raw CER: {summary['avg_raw_cer']:.4f}")
    print(f"Average cleaned CER: {summary['avg_cleaned_cer']:.4f}")
    print(f"Average raw WER: {summary['avg_raw_wer']:.4f}")
    print(f"Average cleaned WER: {summary['avg_cleaned_wer']:.4f}")
    if summary["missing_ground_truth_page_ids"]:
        print(
            f"Skipped {len(summary['missing_ground_truth_page_ids'])} prediction pages with no matching ground truth."
        )


if __name__ == "__main__":
    main()
