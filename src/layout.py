"""Layout heuristics for main-text extraction and line segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils import LayoutConfig


@dataclass
class CropResult:
    """Container for a main text crop."""

    x0: int
    y0: int
    x1: int
    y1: int
    reason: str

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return the crop as an `(x0, y0, x1, y1)` tuple."""

        return self.x0, self.y0, self.x1, self.y1


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive spans where a 1D boolean mask is true."""

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _apply_padding(start: int, end: int, maximum: int, padding: int) -> tuple[int, int]:
    """Expand a span while keeping it within bounds."""

    return max(0, start - padding), min(maximum, end + padding)


def extract_main_text_region(
    gray_image: np.ndarray,
    binary_image: np.ndarray,
    config: LayoutConfig | None = None,
) -> CropResult:
    """Estimate the main printed text block and ignore marginal regions.

    This uses simple row/column text-density projections plus a center bias.
    It is deliberately conservative and intended as a baseline, not a complete
    historical layout parser.
    """

    config = config or LayoutConfig()
    height, width = binary_image.shape
    text_mask = (binary_image > 0).astype(np.uint8)

    row_density = text_mask.sum(axis=1).astype(np.float32)
    if row_density.max() <= 0:
        return _fallback_crop(width, height, config, "fallback_empty_page")

    row_threshold = max(2.0, row_density.max() * config.row_threshold_ratio)
    active_rows = row_density > row_threshold
    row_spans = _contiguous_spans(active_rows)
    if not row_spans:
        return _fallback_crop(width, height, config, "fallback_no_row_span")

    y0, y1 = max(
        row_spans,
        key=lambda span: (span[1] - span[0]) * float(row_density[span[0] : span[1]].mean()),
    )
    y0, y1 = _apply_padding(y0, y1, height, config.row_padding)

    band_mask = text_mask[y0:y1, :]
    col_density = band_mask.sum(axis=0).astype(np.float32)
    if col_density.max() <= 0:
        return _fallback_crop(width, height, config, "fallback_empty_band")

    col_threshold = max(2.0, col_density.max() * config.col_threshold_ratio)
    active_cols = col_density > col_threshold
    col_spans = _contiguous_spans(active_cols)
    if not col_spans:
        return _fallback_crop(width, height, config, "fallback_no_col_span")

    mid_x = width / 2.0

    def span_score(span: tuple[int, int]) -> float:
        left, right = span
        span_width = right - left
        density_score = float(col_density[left:right].mean())
        center = (left + right) / 2.0
        center_distance = abs(center - mid_x) / max(mid_x, 1.0)
        center_bonus = 1.0 - (center_distance * config.center_bias_weight)
        return span_width * density_score * max(center_bonus, 0.1)

    x0, x1 = max(col_spans, key=span_score)
    x0, x1 = _apply_padding(x0, x1, width, config.col_padding)

    if (x1 - x0) < width * 0.2 or (y1 - y0) < height * 0.2:
        return _fallback_crop(width, height, config, "fallback_small_crop")

    return CropResult(x0=x0, y0=y0, x1=x1, y1=y1, reason="projection_crop")


def _fallback_crop(width: int, height: int, config: LayoutConfig, reason: str) -> CropResult:
    """Return a centered fallback crop for pages that defeat the heuristic."""

    crop_h = int(height * config.fallback_height_ratio)
    crop_w = int(width * config.fallback_width_ratio)
    y0 = max(0, (height - crop_h) // 2)
    x0 = max(0, (width - crop_w) // 2)
    return CropResult(x0=x0, y0=y0, x1=x0 + crop_w, y1=y0 + crop_h, reason=reason)


def crop_array(image: np.ndarray, crop: CropResult) -> np.ndarray:
    """Crop a 2D image array using a `CropResult`."""

    return image[crop.y0 : crop.y1, crop.x0 : crop.x1]


def segment_lines(
    gray_crop: np.ndarray,
    binary_crop: np.ndarray,
    config: LayoutConfig | None = None,
) -> list[np.ndarray]:
    """Segment a main text crop into line images using row projection peaks."""

    config = config or LayoutConfig()
    text_mask = (binary_crop > 0).astype(np.uint8)
    row_density = text_mask.sum(axis=1).astype(np.float32)

    if row_density.max() <= 0:
        return [gray_crop]

    threshold = max(1.0, row_density.max() * config.line_threshold_ratio)
    active_rows = row_density > threshold
    spans = _contiguous_spans(active_rows)

    lines: list[np.ndarray] = []
    height = gray_crop.shape[0]
    for start, end in spans:
        if (end - start) < config.min_line_height:
            continue
        y0, y1 = _apply_padding(start, end, height, config.line_padding)
        line = gray_crop[y0:y1, :]
        if line.size:
            lines.append(line)

    return lines or [gray_crop]
