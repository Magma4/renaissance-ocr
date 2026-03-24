from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass
class PreprocessConfig:
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    median_blur_kernel: int = 3
    adaptive_block_size: int = 41
    adaptive_c: int = 15
    morph_kernel_size: int = 2


@dataclass
class LayoutConfig:
    row_threshold_ratio: float = 0.15
    col_threshold_ratio: float = 0.20
    row_padding: int = 20
    col_padding: int = 30
    center_bias_weight: float = 0.35
    fallback_height_ratio: float = 0.92
    fallback_width_ratio: float = 0.82
    line_threshold_ratio: float = 0.12
    min_line_height: int = 12
    line_padding: int = 4


@dataclass
class OCRConfig:
    model_name: str = "microsoft/trocr-base-printed"
    batch_size: int = 8
    max_new_tokens: int = 128
    num_beams: int = 1
    device: str | None = None


@dataclass
class CRNNConfig:
    """Configuration for the CRNN OCR model."""

    checkpoint_path: str | None = None
    vocab_path: str | None = None
    device: str | None = None
    decoder: str = "greedy"  # "greedy" | "beam"
    beam_width: int = 8
    lexicon_path: str | None = None
    lexicon_bonus: float = 2.0


@dataclass
class CleanupConfig:
    """backend: 'rule_based' | 'manual' | 'gemini'"""

    backend: str = "rule_based"
    manual_prompt_dir: Path | None = None
    manual_response_dir: Path | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "models/gemini-flash-lite-latest"
    safe_replacements: dict[str, str] = field(
        default_factory=lambda: {
            "\u00ad": "",
            "\ufb01": "fi",
            "\ufb02": "fl",
        }
    )


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def save_text(path: Path | str, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path | str) -> Any:
    return json.loads(load_text(path))


def save_json(path: Path | str, payload: Any, indent: int = 2) -> None:
    save_text(path, json.dumps(payload, indent=indent, ensure_ascii=False))


def read_jsonl(path: Path | str, missing_ok: bool = False) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"Could not find JSONL file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_by(records: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        if value in index:
            raise ValueError(f"Duplicate {key!r} value: {value}")
        index[str(value)] = record
    return index


def slugify_name(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered)
    return cleaned.strip("_") or "document"


def make_page_id(source_name: str, page_number: int) -> str:
    return f"{slugify_name(source_name)}_page_{page_number:04d}"


def sorted_paths(paths: Iterable[Path]) -> list[Path]:
    def key(path: Path) -> list[Any]:
        parts = re.split(r"(\d+)", path.name.lower())
        return [int(part) if part.isdigit() else part for part in parts]

    return sorted(paths, key=key)


def list_image_files(input_dir: Path | str) -> list[Path]:
    root = Path(input_dir)
    candidates: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
        candidates.extend(root.glob(pattern))
    return sorted_paths(candidates)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_metrics(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
