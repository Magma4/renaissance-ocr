"""OCR model wrapper for a zero-shot printed-text TrOCR baseline."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.utils import OCRConfig


class TrOCRPrintedBaseline:
    """Thin wrapper around a Hugging Face TrOCR printed-text checkpoint."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        if self.config.device is None:
            self.config = replace(
                self.config,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        self.processor = TrOCRProcessor.from_pretrained(self.config.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_lines(self, line_images: Iterable[Image.Image]) -> list[str]:
        """OCR a batch of line images."""

        images = [image.convert("RGB") for image in line_images]
        if not images:
            return []

        outputs: list[str] = []
        batch_size = max(self.config.batch_size, 1)
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.config.device)
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
            )
            predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            outputs.extend([prediction.strip() for prediction in predictions])
        return outputs

    def predict_page_text(self, line_images: Iterable[Image.Image]) -> str:
        """OCR line images and join them into a page transcript."""

        lines = self.predict_lines(line_images)
        return "\n".join(line for line in lines if line)
