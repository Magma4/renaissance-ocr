"""CNN-RNN (CRNN) model for historical printed text OCR with weighted CTC loss.

Architecture overview
---------------------
1. CNN backbone  – four conv blocks that collapse the image height to 1 while
                   progressively widening the channel dimension.
2. BiLSTM head   – two bidirectional LSTM layers treating the remaining width
                   dimension as a time sequence.
3. Linear proj   – maps LSTM hidden states to vocabulary logits.

Training uses CTC loss.  To handle the pronounced class imbalance in 17th-century
Spanish text (e.g. rare diacritics vs. common letters) the trainer computes per-
character inverse-frequency weights and passes them to a weighted variant of CTC.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.utils import CRNNConfig


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

BLANK_TOKEN = "<BLANK>"
UNKNOWN_TOKEN = "<UNK>"


def build_vocab(texts: Iterable[str]) -> dict[str, int]:
    """Build char-to-index vocabulary from an iterable of training strings."""
    chars: set[str] = set()
    for text in texts:
        chars.update(text)
    sorted_chars = sorted(chars)
    vocab: dict[str, int] = {BLANK_TOKEN: 0}
    for i, ch in enumerate(sorted_chars, start=1):
        vocab[ch] = i
    vocab[UNKNOWN_TOKEN] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: Path | str) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_char_weights(
    texts: Iterable[str],
    vocab: dict[str, int],
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Return per-class inverse-frequency weights (length = len(vocab)).

    Rare characters (diacritics, ligatures) get higher weight so the CTC loss
    penalises their misrecognition more.  The blank token always gets weight 1.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(text)

    weights = torch.ones(len(vocab))
    total = sum(counts.values()) + smoothing * len(vocab)
    for ch, idx in vocab.items():
        if ch in (BLANK_TOKEN, UNKNOWN_TOKEN):
            continue
        freq = (counts.get(ch, 0) + smoothing) / total
        weights[idx] = 1.0 / freq

    # Normalise so the mean weight is 1
    weights = weights / weights.mean()
    return weights


# ---------------------------------------------------------------------------
# CNN backbone
# ---------------------------------------------------------------------------

def _conv_block(
    in_ch: int,
    out_ch: int,
    *,
    pool: tuple[int, int] | None = (2, 2),
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    if pool is not None:
        layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


class CNNBackbone(nn.Module):
    """Four-block CNN that maps (B, 1, H, W) → (B, 512, 1, W').

    Uses AdaptiveAvgPool on the last block to guarantee height→1 regardless
    of the exact input height.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _conv_block(1, 64, pool=(2, 2)),
            _conv_block(64, 128, pool=(2, 2)),
            _conv_block(128, 256, pool=None),
            _conv_block(256, 512, pool=None),
        )
        # Collapse height to exactly 1 after the conv stack
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        x = self.blocks(x)          # (B, 512, H', W')
        x = self.height_pool(x)     # (B, 512, 1, W')
        x = x.squeeze(2)            # (B, 512, W')
        return x.permute(2, 0, 1)   # (W', B, 512)


# ---------------------------------------------------------------------------
# Full CRNN
# ---------------------------------------------------------------------------

class CRNN(nn.Module):
    """Convolutional-Recurrent network for CTC-based sequence recognition."""

    def __init__(self, num_classes: int, hidden_size: int = 256, num_lstm_layers: int = 2) -> None:
        super().__init__()
        self.cnn = CNNBackbone()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        features = self.cnn(x)              # (T, B, 512)
        recurrent, _ = self.lstm(features)  # (T, B, 2*hidden)
        logits = self.classifier(recurrent) # (T, B, num_classes)
        return F.log_softmax(logits, dim=2)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class CRNNOCRModel:
    """High-level wrapper: image → recognised text string."""

    def __init__(self, config: CRNNConfig | None = None) -> None:
        self.config = config or CRNNConfig()
        if self.config.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.config = replace(self.config, device=device)

        self.vocab: dict[str, int] = {}
        self.idx_to_char: dict[int, str] = {}
        self.model: CRNN | None = None

        if self.config.vocab_path and Path(self.config.vocab_path).exists():
            self._load_vocab(Path(self.config.vocab_path))

        if self.config.checkpoint_path and Path(self.config.checkpoint_path).exists():
            self._load_checkpoint(Path(self.config.checkpoint_path))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_vocab(self, path: Path) -> None:
        self.vocab = load_vocab(path)
        self.idx_to_char = {v: k for k, v in self.vocab.items()}

    def _load_checkpoint(self, path: Path) -> None:
        if not self.vocab:
            raise RuntimeError("Load vocabulary before checkpoint.")
        num_classes = len(self.vocab)
        self.model = CRNN(num_classes=num_classes)
        state = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(state)
        self.model.to(self.config.device)
        self.model.eval()

    def build_from_vocab(self, vocab: dict[str, int]) -> None:
        """Initialise a fresh CRNN from a vocabulary dict (for training)."""
        self.vocab = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.model = CRNN(num_classes=len(vocab))
        self.model.to(self.config.device)

    # ------------------------------------------------------------------
    # Image pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(image: Image.Image, target_height: int = 32) -> torch.Tensor:
        """Resize to fixed height, convert to grayscale tensor (B=1, C=1, H, W)."""
        grayscale = image.convert("L")
        w, h = grayscale.size
        new_w = max(1, int(w * target_height / h))
        resized = grayscale.resize((new_w, target_height), Image.LANCZOS)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return tensor

    # ------------------------------------------------------------------
    # Greedy CTC decode
    # ------------------------------------------------------------------

    def _greedy_decode(self, log_probs: torch.Tensor) -> str:
        """Greedy CTC decode: argmax → collapse repeats → remove blank."""
        indices = log_probs.argmax(dim=2).squeeze(1).tolist()  # (T,)
        result: list[str] = []
        prev: int | None = None
        blank_idx = self.vocab.get(BLANK_TOKEN, 0)
        for idx in indices:
            if idx != prev and idx != blank_idx:
                ch = self.idx_to_char.get(idx, "")
                if ch and ch != UNKNOWN_TOKEN:
                    result.append(ch)
            prev = idx
        return "".join(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_lines(self, line_images: Iterable[Image.Image]) -> list[str]:
        """OCR a list of line images; returns a string per line."""
        if self.model is None:
            raise RuntimeError("Model not loaded.  Call build_from_vocab() or load a checkpoint.")
        results: list[str] = []
        self.model.eval()
        with torch.inference_mode():
            for image in line_images:
                tensor = self._prepare_image(image).to(self.config.device)
                log_probs = self.model(tensor)  # (T, 1, num_classes)
                text = self._greedy_decode(log_probs)
                results.append(text)
        return results

    def predict_page_text(self, line_images: Iterable[Image.Image]) -> str:
        """OCR line images and join into a page transcript."""
        lines = self.predict_lines(line_images)
        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def get_char_weights(
        texts: Iterable[str],
        vocab: dict[str, int],
        smoothing: float = 1.0,
    ) -> torch.Tensor:
        """Public alias for :func:`compute_char_weights`."""
        return compute_char_weights(texts, vocab, smoothing=smoothing)
