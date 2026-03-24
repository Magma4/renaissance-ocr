"""CTC decoding strategies for the CRNN OCR model.

Two decoders are provided:

1. ``GreedyCTCDecoder``   – simple argmax-collapse with blank removal.
2. ``LexiconBeamSearchDecoder`` – a pure-Python prefix beam search with
   optional lexicon rescoring.  It does not require external C extensions
   so it works without ``pyctcdecode`` or KenLM installed.

Usage
-----
    from src.decoding import GreedyCTCDecoder, LexiconBeamSearchDecoder

    decoder = GreedyCTCDecoder(vocab)
    text = decoder.decode(log_probs)           # log_probs: (T, 1, C) tensor

    lex_decoder = LexiconBeamSearchDecoder(vocab, lexicon_path="data/lexicon/...")
    text = lex_decoder.decode(log_probs)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch

BLANK_TOKEN = "<BLANK>"
UNKNOWN_TOKEN = "<UNK>"


# ---------------------------------------------------------------------------
# Greedy decoder
# ---------------------------------------------------------------------------

class GreedyCTCDecoder:
    """Argmax CTC decoder.  Fast and sufficient for most line images."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.idx_to_char: dict[int, str] = {v: k for k, v in vocab.items()}
        self.blank_idx: int = vocab.get(BLANK_TOKEN, 0)

    def decode(self, log_probs: torch.Tensor) -> str:
        """
        Parameters
        ----------
        log_probs : (T, B, C) log-softmax tensor – only the first batch item
                    is decoded.
        """
        indices = log_probs[:, 0, :].argmax(dim=1).tolist()
        result: list[str] = []
        prev: int | None = None
        for idx in indices:
            if idx != prev and idx != self.blank_idx:
                ch = self.idx_to_char.get(idx, "")
                if ch and ch not in (UNKNOWN_TOKEN,):
                    result.append(ch)
            prev = idx
        return "".join(result)

    def decode_batch(self, log_probs: torch.Tensor) -> list[str]:
        """Decode all items in a batch.  log_probs: (T, B, C)."""
        results: list[str] = []
        for b in range(log_probs.size(1)):
            indices = log_probs[:, b, :].argmax(dim=1).tolist()
            text_chars: list[str] = []
            prev: int | None = None
            for idx in indices:
                if idx != prev and idx != self.blank_idx:
                    ch = self.idx_to_char.get(idx, "")
                    if ch and ch not in (UNKNOWN_TOKEN,):
                        text_chars.append(ch)
                prev = idx
            results.append("".join(text_chars))
        return results


# ---------------------------------------------------------------------------
# Lexicon-constrained beam search decoder
# ---------------------------------------------------------------------------

def _load_lexicon(path: Path | str) -> set[str]:
    """Load one word per line, lowercase, strip whitespace."""
    p = Path(path)
    if not p.exists():
        return set()
    words: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word:
            words.add(word)
    return words


class _BeamEntry:
    __slots__ = ("text", "p_b", "p_nb")

    def __init__(self, text: str = "", p_b: float = 0.0, p_nb: float = float("-inf")) -> None:
        self.text = text
        self.p_b = p_b      # log-prob of last token being blank
        self.p_nb = p_nb    # log-prob of last token being non-blank

    @property
    def total(self) -> float:
        return math.log(math.exp(self.p_b) + math.exp(self.p_nb) + 1e-30)


class LexiconBeamSearchDecoder:
    """Prefix beam search with optional lexicon word-boundary rescoring.

    The lexicon rescoring adds a small bonus to complete words that appear in
    the provided wordlist, which nudges the decoder toward period-correct
    Spanish orthography without completely blocking OOV words.

    Parameters
    ----------
    vocab         : char → index mapping from CRNN training
    lexicon_path  : path to one-word-per-line lexicon file (optional)
    beam_width    : number of active prefixes per step
    lexicon_bonus : log-space bonus for completing a known word
    """

    def __init__(
        self,
        vocab: dict[str, int],
        lexicon_path: Path | str | None = None,
        beam_width: int = 8,
        lexicon_bonus: float = 2.0,
    ) -> None:
        self.idx_to_char: dict[int, str] = {v: k for k, v in vocab.items()}
        self.blank_idx = vocab.get(BLANK_TOKEN, 0)
        self.beam_width = beam_width
        self.lexicon_bonus = lexicon_bonus
        self.lexicon: set[str] = set()
        if lexicon_path:
            self.lexicon = _load_lexicon(lexicon_path)

    # ------------------------------------------------------------------

    def _word_score(self, text: str) -> float:
        """Return lexicon bonus if the last word in text is in the lexicon."""
        if not self.lexicon or not text:
            return 0.0
        words = text.lower().split()
        if not words:
            return 0.0
        last_word = "".join(c for c in words[-1] if c.isalpha())
        return self.lexicon_bonus if last_word in self.lexicon else 0.0

    # ------------------------------------------------------------------

    def decode(self, log_probs: torch.Tensor) -> str:
        """
        Parameters
        ----------
        log_probs : (T, B, C) tensor – only the first batch item is decoded.
        """
        T = log_probs.size(0)
        C = log_probs.size(2)
        probs = log_probs[:, 0, :].cpu().float()  # (T, C)

        # Initialise beam with empty prefix
        beams: dict[str, _BeamEntry] = {"": _BeamEntry(p_b=0.0)}

        for t in range(T):
            new_beams: dict[str, _BeamEntry] = {}
            step = probs[t]  # (C,)

            # Sort by total probability to prune efficiently
            top_prefs = sorted(beams.values(), key=lambda e: e.total, reverse=True)[: self.beam_width]

            for entry in top_prefs:
                prefix = entry.text

                # --- extend with blank ---
                log_p_blank = step[self.blank_idx].item()
                new_p_b = math.log(math.exp(entry.p_b) + math.exp(entry.p_nb) + 1e-30) + log_p_blank
                if prefix not in new_beams:
                    new_beams[prefix] = _BeamEntry(text=prefix)
                new_beams[prefix].p_b = math.log(
                    math.exp(new_beams[prefix].p_b) + math.exp(new_p_b) + 1e-30
                )

                # --- extend with each character ---
                for idx in range(C):
                    if idx == self.blank_idx:
                        continue
                    ch = self.idx_to_char.get(idx, "")
                    if not ch or ch == UNKNOWN_TOKEN:
                        continue
                    log_p_c = step[idx].item()
                    new_prefix = prefix + ch

                    # Avoid extending with same char if last was non-blank
                    if prefix and prefix[-1] == ch:
                        new_p_nb = entry.p_b + log_p_c
                    else:
                        new_p_nb = math.log(
                            math.exp(entry.p_b) + math.exp(entry.p_nb) + 1e-30
                        ) + log_p_c

                    # Lexicon rescoring at word boundaries
                    if ch == " ":
                        new_p_nb += self._word_score(prefix)

                    if new_prefix not in new_beams:
                        new_beams[new_prefix] = _BeamEntry(text=new_prefix)
                    new_beams[new_prefix].p_nb = math.log(
                        math.exp(new_beams[new_prefix].p_nb) + math.exp(new_p_nb) + 1e-30
                    )

            # Prune to beam_width
            beams = dict(
                sorted(new_beams.items(), key=lambda kv: kv[1].total, reverse=True)[: self.beam_width]
            )

        best = max(beams.values(), key=lambda e: e.total)
        return best.text.strip()
