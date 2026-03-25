from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

# Disable PIL DOS protection since historical manuscript scans are massive
Image.MAX_IMAGE_PIXELS = None


try:
    import google.genai as genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


VLM_OCR_PROMPT = """You are an expert paleographer specializing in early modern (17th-century) Spanish handwritten documents. 
Your task is to provide an exact, highly accurate transcription of the text contained in the provided image.

Instructions:
1. Transcribe the manuscript exactly as written.
2. Resolve standard period abbreviations (e.g., expanding tildes over letters if obvious).
3. Preserve the original capitalization and punctuation where legible.
4. Correct obvious spelling errors only if they are clearly grammatical mistakes of the author, but generally default to the exact historical spelling on the page.
5. Return ONLY the transcribed text. Do not include introductory remarks or explanations.
"""

class GeminiVLMExtractor:
    """End-to-end OCR pipeline using Gemini 2.5 Flash as a Vision-Language Model.
    
    This reads an image, passes it directly to Gemini along with a specialized
    paleography prompt, and returns the transcribed text.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        if not GEMINI_AVAILABLE:
            raise ImportError("The 'google-genai' package is required but not installed.")
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def extract_text(self, image_path: Path | str) -> str:
        """Transcribe text from a handwritten manuscript image in a single zero-shot pass."""
        image = Image.open(image_path)
        
        # Downscale massive manuscript images to prevent Gemini payload overflow
        max_dim = 3000
        if max(image.width, image.height) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                image, 
                VLM_OCR_PROMPT
            ]
        )
        
        if not response.text:
            raise RuntimeError("Gemini API returned an empty response.")
        
        return response.text.strip()
