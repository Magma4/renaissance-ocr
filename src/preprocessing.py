"""Page preprocessing utilities for scanned historical print."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from src.utils import PreprocessConfig


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to OpenCV BGR format."""

    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL image."""

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def grayscale_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a single-channel array to a PIL image."""

    return Image.fromarray(image)


def preprocess_page(
    image: Image.Image,
    config: PreprocessConfig | None = None,
) -> dict[str, np.ndarray]:
    """Generate grayscale and binarized views of a page image.

    The grayscale view is used for OCR crops, while the binary view is used
    for rough layout analysis and line segmentation.
    """

    config = config or PreprocessConfig()
    bgr = pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if config.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_grid_size,
        )
        gray = clahe.apply(gray)

    if config.median_blur_kernel > 1:
        kernel = config.median_blur_kernel
        if kernel % 2 == 0:
            kernel += 1
        gray = cv2.medianBlur(gray, kernel)

    block_size = config.adaptive_block_size
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        config.adaptive_c,
    )

    if config.morph_kernel_size > 0:
        kernel = np.ones(
            (config.morph_kernel_size, config.morph_kernel_size),
            dtype=np.uint8,
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return {"gray": gray, "binary": binary}


def prepare_line_for_ocr(line_image: np.ndarray, border: int = 8) -> Image.Image:
    """Pad a grayscale line crop before OCR."""

    line = line_image
    if line.ndim != 2:
        raise ValueError("Expected a single-channel line image.")

    line = np.pad(
        line,
        ((border, border), (border, border)),
        mode="constant",
        constant_values=255,
    )
    return grayscale_to_pil(line)


def update_preprocess_config(**kwargs: object) -> PreprocessConfig:
    """Convenience helper for notebook-driven config tweaks."""

    return replace(PreprocessConfig(), **kwargs)
