from __future__ import annotations

import os
import re
from pathlib import Path

from src.utils import CleanupConfig, ensure_dir, load_text, normalize_whitespace, save_text


HISTORICAL_OCR_PROMPT_TEMPLATE = """You are correcting OCR output from 17th-century printed Spanish text.

Preserve meaning and historical spelling where reasonable.
Fix obvious OCR mistakes only.
Do not modernize the spelling unless the OCR is clearly wrong.
Do not rewrite the passage.
Return cleaned transcription only.

OCR output:
{ocr_text}
"""


class RuleBasedCleaner:
    def __init__(self, config: CleanupConfig | None = None) -> None:
        self.config = config or CleanupConfig()

    def clean(self, text: str) -> str:
        cleaned = text
        for source, target in self.config.safe_replacements.items():
            cleaned = cleaned.replace(source, target)

        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", cleaned)
        cleaned = re.sub(r" *\n *", "\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([\(\[{])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s+([\)\]}])", r"\1", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return normalize_whitespace(cleaned)


class ManualPromptCleaner:
    def __init__(self, config: CleanupConfig | None = None) -> None:
        self.config = config or CleanupConfig()

    def build_prompt(self, raw_text: str) -> str:
        return HISTORICAL_OCR_PROMPT_TEMPLATE.format(ocr_text=raw_text.strip())

    def write_prompt(self, page_id: str, raw_text: str) -> Path | None:
        if not self.config.manual_prompt_dir:
            return None

        prompt_dir = ensure_dir(self.config.manual_prompt_dir)
        prompt_path = prompt_dir / f"{page_id}.prompt.txt"
        save_text(prompt_path, self.build_prompt(raw_text))
        return prompt_path

    def read_response(self, page_id: str) -> str | None:
        if not self.config.manual_response_dir:
            return None

        response_path = Path(self.config.manual_response_dir) / f"{page_id}.txt"
        if not response_path.exists():
            return None
        return load_text(response_path).strip()


class GeminiCleaner:
    """Post-OCR cleanup using the Google Gemini API.

    Requires the ``google-generativeai`` package and a valid ``GEMINI_API_KEY``
    environment variable (or the key passed directly in ``CleanupConfig``).

    If the API key is absent or the call fails the cleaner falls back
    transparently to the rule-based backend.
    """

    def __init__(self, config: CleanupConfig | None = None) -> None:
        self.config = config or CleanupConfig()
        self._client = None
        self._available = False
        self._setup()

    def _setup(self) -> None:
        api_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return
        try:
            import google.genai as genai  # type: ignore[import]

            self._client = genai.Client(api_key=api_key)
            self._model_name = self.config.gemini_model
            self._available = True
        except Exception:
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def clean(self, raw_text: str) -> tuple[str, str]:
        """Return (cleaned_text, backend_used)."""
        if not self._available or not raw_text.strip():
            rule_cleaner = RuleBasedCleaner(self.config)
            return rule_cleaner.clean(raw_text), "rule_based_fallback"

        prompt = HISTORICAL_OCR_PROMPT_TEMPLATE.format(ocr_text=raw_text.strip())
        try:
            import google.genai as genai  # type: ignore[import]

            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self._model_name,
                contents=prompt,
            )
            cleaned = response.text.strip() if response.text else ""
            if not cleaned:
                raise ValueError("Empty response from Gemini.")
            return cleaned, "gemini"
        except Exception:
            rule_cleaner = RuleBasedCleaner(self.config)
            return rule_cleaner.clean(raw_text), "rule_based_fallback"


def run_cleanup(page_id: str, raw_text: str, config: CleanupConfig | None = None) -> dict[str, str | None]:
    config = config or CleanupConfig()
    rule_cleaner = RuleBasedCleaner(config)
    fallback_text = rule_cleaner.clean(raw_text)

    # ── rule_based ──────────────────────────────────────────────────────
    if config.backend == "rule_based":
        return {
            "cleaned_text": fallback_text,
            "backend_requested": "rule_based",
            "backend_used": "rule_based",
            "prompt_path": None,
        }

    # ── gemini ──────────────────────────────────────────────────────────
    if config.backend == "gemini":
        gemini = GeminiCleaner(config)
        cleaned, backend_used = gemini.clean(raw_text)
        return {
            "cleaned_text": cleaned,
            "backend_requested": "gemini",
            "backend_used": backend_used,
            "prompt_path": None,
        }

    # ── manual ──────────────────────────────────────────────────────────
    if config.backend != "manual":
        raise ValueError(f"Unsupported cleanup backend: {config.backend!r}")

    manual_cleaner = ManualPromptCleaner(config)
    prompt_path = manual_cleaner.write_prompt(page_id, raw_text)
    manual_response = manual_cleaner.read_response(page_id)

    if manual_response:
        return {
            "cleaned_text": manual_response,
            "backend_requested": "manual",
            "backend_used": "manual_response",
            "prompt_path": str(prompt_path) if prompt_path else None,
        }

    return {
        "cleaned_text": fallback_text,
        "backend_requested": "manual",
        "backend_used": "rule_based_fallback",
        "prompt_path": str(prompt_path) if prompt_path else None,
    }
