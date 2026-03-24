"""Train the CRNN OCR model on extracted line crops and ground-truth text.

The script:
  1. Parses the ground-truth JSONL file.
  2. Locates line crop images under --line-image-dir (written by run_baseline.py).
     Each crop is named  <page_id>_line_<N>.png  OR the script can fall back to
     using the full page crop <page_id>_crop.png as a single-image sample.
  3. Builds a character vocabulary from all ground-truth text.
  4. Computes per-character inverse-frequency weights for the CTC loss.
  5. Trains the CRNN with AdamW + cosine LR schedule.
  6. Saves the best checkpoint and vocabulary.

Usage
-----
    python scripts/train_crnn.py \\
        --ground-truth-file data/ground_truth/ground_truth.jsonl \\
        --line-image-dir data/crops \\
        --output-dir outputs/crnn \\
        --epochs 30

For a quick smoke test:
    python scripts/train_crnn.py \\
        --ground-truth-file data/ground_truth/ground_truth.jsonl \\
        --line-image-dir data/crops \\
        --output-dir outputs/crnn \\
        --epochs 1 --limit 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.crnn_model import (
    CRNN,
    CRNNOCRModel,
    build_vocab,
    compute_char_weights,
    save_vocab,
)
from src.utils import CRNNConfig, ensure_dir, read_jsonl


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CROPLineDataset(Dataset):
    """Dataset of (image_tensor, label_indices, label_length) triples.

    If per-line crops (<page_id>_line_<N>.png) are not found, falls back to
    the full page crop (<page_id>_crop.png) paired with the full page text.
    """

    TARGET_HEIGHT = 32

    def __init__(
        self,
        records: list[dict],
        image_dir: Path,
        vocab: dict[str, int],
        limit: int | None = None,
    ) -> None:
        self.vocab = vocab
        self.samples: list[tuple[Path, str]] = []

        for record in records:
            page_id = record["page_id"]
            text = record["text"].strip()
            if not text:
                continue

            # Try to find per-line crops first
            line_crops = sorted(image_dir.glob(f"{page_id}_line_*.png"))
            if line_crops:
                # Pair each line crop with the corresponding text line
                text_lines = [ln for ln in text.splitlines() if ln.strip()]
                for img_path, line_text in zip(line_crops, text_lines):
                    if line_text.strip():
                        self.samples.append((img_path, line_text.strip()))
            else:
                # Fall back to full-page crop image
                crop_path = image_dir / f"{page_id}_crop.png"
                if crop_path.exists():
                    self.samples.append((crop_path, text))

        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("L")

        # Resize to fixed height while preserving aspect ratio
        w, h = image.size
        new_w = max(1, int(w * self.TARGET_HEIGHT / h))
        image = image.resize((new_w, self.TARGET_HEIGHT), Image.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

        unknown_idx = self.vocab.get("<UNK>", 0)
        label = torch.tensor(
            [self.vocab.get(ch, unknown_idx) for ch in text],
            dtype=torch.long,
        )
        return tensor, label, len(label)


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad images to max width; stack labels."""
    images, labels, label_lengths = zip(*batch)

    max_w = max(img.shape[-1] for img in images)
    padded_images = torch.zeros(len(images), 1, images[0].shape[-2], max_w)
    for i, img in enumerate(images):
        padded_images[i, :, :, : img.shape[-1]] = img

    label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long)
    labels_concat = torch.cat(labels)
    return padded_images, labels_concat, label_lengths_tensor, label_lengths_tensor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: CRNN,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, label_lengths, input_lengths_hint in loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        log_probs = model(images)  # (T, B, C)
        T = log_probs.size(0)
        B = log_probs.size(1)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CRNN OCR model.")
    parser.add_argument("--ground-truth-file", type=Path, required=True)
    parser.add_argument(
        "--line-image-dir",
        type=Path,
        default=Path("data/crops"),
        help="Directory containing <page_id>_crop.png or <page_id>_line_N.png files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/crnn"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None, help="Cap dataset size (for testing).")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-lstm-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    records = read_jsonl(args.ground_truth_file)
    if not records:
        raise SystemExit("Ground truth file is empty.  Run parse_ground_truth.py first.")

    texts = [r["text"] for r in records]
    vocab = build_vocab(texts)
    vocab_path = output_dir / "crnn_vocab.json"
    save_vocab(vocab, vocab_path)
    print(f"Vocabulary size: {len(vocab)} characters → {vocab_path}")

    char_weights = compute_char_weights(texts, vocab).to(device)

    dataset = CROPLineDataset(records, args.line_image_dir, vocab, limit=args.limit)
    if not dataset.samples:
        raise SystemExit(
            f"No image-text pairs found.  Make sure crops exist in {args.line_image_dir}."
        )
    print(f"Dataset size: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = CRNN(
        num_classes=len(vocab),
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
    ).to(device)

    criterion = nn.CTCLoss(blank=vocab["<BLANK>"], zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_loss = float("inf")
    checkpoint_path = output_dir / "crnn_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimiser, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nTraining complete.  Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Vocabulary saved to: {vocab_path}")

    # Save a small training summary
    summary = {
        "epochs": args.epochs,
        "dataset_size": len(dataset),
        "vocab_size": len(vocab),
        "best_loss": best_loss,
        "checkpoint": str(checkpoint_path),
        "vocab_path": str(vocab_path),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
