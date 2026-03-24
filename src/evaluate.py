from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import (
    ensure_dir,
    index_by,
    normalize_for_metrics,
    read_jsonl,
    save_json,
    save_text,
    write_jsonl,
)


def _levenshtein_distance(source: list[str], target: list[str]) -> int:
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_token in enumerate(source, start=1):
        current = [i]
        for j, target_token in enumerate(target, start=1):
            substitution_cost = 0 if source_token == target_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    reference_chars = list(normalize_for_metrics(reference))
    hypothesis_chars = list(normalize_for_metrics(hypothesis))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return _levenshtein_distance(reference_chars, hypothesis_chars) / len(reference_chars)


def compute_wer(reference: str, hypothesis: str) -> float:
    reference_words = normalize_for_metrics(reference).split()
    hypothesis_words = normalize_for_metrics(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _levenshtein_distance(reference_words, hypothesis_words) / len(reference_words)


def load_ground_truth_records(path: Path | str) -> list[dict[str, str]]:
    records = read_jsonl(path)
    cleaned_records: list[dict[str, str]] = []

    for index, record in enumerate(records, start=1):
        page_id = record.get("page_id")
        text = record.get("text")
        if not page_id or text is None:
            raise ValueError(
                f"Ground truth record {index} is missing 'page_id' or 'text'."
            )
        cleaned_records.append(
            {
                "page_id": str(page_id),
                "text": str(text).strip(),
            }
        )

    index_by(cleaned_records, "page_id")
    return cleaned_records


def evaluate_records(
    prediction_records: list[dict[str, Any]],
    ground_truth_records: list[dict[str, str]],
) -> dict[str, Any]:
    ground_truth_by_page = index_by(ground_truth_records, "page_id")
    comparisons: list[dict[str, Any]] = []
    missing_ground_truth: list[str] = []

    for record in prediction_records:
        page_id = str(record.get("page_id", "")).strip()
        if not page_id:
            continue
        ground_truth_record = ground_truth_by_page.get(page_id)
        if ground_truth_record is None:
            missing_ground_truth.append(page_id)
            continue

        ground_truth = ground_truth_record["text"]
        raw_ocr = str(record.get("raw_ocr", "") or "")
        cleaned_ocr = str(record.get("cleaned_ocr", "") or "")
        raw_cer = compute_cer(ground_truth, raw_ocr)
        cleaned_cer = compute_cer(ground_truth, cleaned_ocr)
        raw_wer = compute_wer(ground_truth, raw_ocr)
        cleaned_wer = compute_wer(ground_truth, cleaned_ocr)

        comparisons.append(
            {
                "page_id": page_id,
                "ground_truth": ground_truth,
                "raw_ocr": raw_ocr,
                "cleaned_ocr": cleaned_ocr,
                "raw_cer": raw_cer,
                "cleaned_cer": cleaned_cer,
                "raw_wer": raw_wer,
                "cleaned_wer": cleaned_wer,
                "cer_delta": cleaned_cer - raw_cer,
                "wer_delta": cleaned_wer - raw_wer,
            }
        )

    evaluated_page_ids = {item["page_id"] for item in comparisons}
    unused_ground_truth = [
        record["page_id"]
        for record in ground_truth_records
        if record["page_id"] not in evaluated_page_ids
    ]

    def average(metric_name: str) -> float:
        if not comparisons:
            return 0.0
        return sum(item[metric_name] for item in comparisons) / len(comparisons)

    cleanup_helped_cer = sum(1 for item in comparisons if item["cleaned_cer"] < item["raw_cer"])
    cleanup_hurt_cer = sum(1 for item in comparisons if item["cleaned_cer"] > item["raw_cer"])

    return {
        "summary": {
            "prediction_pages": len(prediction_records),
            "ground_truth_pages": len(ground_truth_records),
            "evaluated_pages": len(comparisons),
            "missing_ground_truth_page_ids": sorted(set(missing_ground_truth)),
            "unused_ground_truth_page_ids": unused_ground_truth,
            "avg_raw_cer": average("raw_cer"),
            "avg_cleaned_cer": average("cleaned_cer"),
            "avg_raw_wer": average("raw_wer"),
            "avg_cleaned_wer": average("cleaned_wer"),
            "cleanup_helped_pages_cer": cleanup_helped_cer,
            "cleanup_hurt_pages_cer": cleanup_hurt_cer,
        },
        "comparisons": comparisons,
    }


def save_evaluation_artifacts(
    evaluation: dict[str, Any],
    output_dir: Path | str,
    example_limit: int = 5,
) -> None:
    output_dir = ensure_dir(output_dir)
    summary = evaluation["summary"]
    comparisons = evaluation["comparisons"]
    sample_examples = comparisons[:example_limit]

    save_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "comparisons.jsonl", comparisons)
    write_jsonl(output_dir / "sample_examples.jsonl", sample_examples)

    lines = [
        "# Sample OCR comparisons",
        "",
        f"Evaluated pages: {summary['evaluated_pages']}",
        f"Average raw CER: {summary['avg_raw_cer']:.4f}",
        f"Average cleaned CER: {summary['avg_cleaned_cer']:.4f}",
        f"Average raw WER: {summary['avg_raw_wer']:.4f}",
        f"Average cleaned WER: {summary['avg_cleaned_wer']:.4f}",
        "",
    ]

    for item in sample_examples:
        lines.extend(
            [
                f"## {item['page_id']}",
                "",
                f"- Raw CER: {item['raw_cer']:.4f}",
                f"- Cleaned CER: {item['cleaned_cer']:.4f}",
                f"- Raw WER: {item['raw_wer']:.4f}",
                f"- Cleaned WER: {item['cleaned_wer']:.4f}",
                "",
                "### Ground truth",
                "",
                "```text",
                item["ground_truth"].strip(),
                "```",
                "",
                "### Raw OCR",
                "",
                "```text",
                item["raw_ocr"].strip(),
                "```",
                "",
                "### Cleaned OCR",
                "",
                "```text",
                item["cleaned_ocr"].strip(),
                "```",
                "",
            ]
        )

    save_text(output_dir / "examples.md", "\n".join(lines).rstrip() + "\n")
