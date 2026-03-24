"""Fine-tune microsoft/trocr-base-printed on the extracted ground-truth pages.

Wraps Hugging Face Seq2SeqTrainer so you can run this with a single command.
Even a few pages of ground truth (the 5 we have right now) will shift the
model toward 17th-century Spanish orthography.

Usage
-----
    python scripts/finetune_trocr.py \\
        --ground-truth-file data/ground_truth/ground_truth.jsonl \\
        --image-dir data/crops \\
        --output-dir outputs/trocr_finetuned \\
        --epochs 5

For a very quick sanity check (1 image, 1 step):
    python scripts/finetune_trocr.py \\
        --ground-truth-file data/ground_truth/ground_truth.jsonl \\
        --image-dir data/crops \\
        --output-dir outputs/trocr_finetuned \\
        --epochs 1 --max-steps 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from src.utils import ensure_dir, read_jsonl


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrOCRLineDataset(Dataset):
    """Pairs page-crop images with their ground-truth text.

    Looks for <page_id>_crop.png in image_dir for each record.
    Skips pages with no matching image.
    """

    def __init__(
        self,
        records: list[dict],
        image_dir: Path,
        processor: TrOCRProcessor,
        max_target_length: int = 128,
    ) -> None:
        self.processor = processor
        self.max_target_length = max_target_length
        self.samples: list[tuple[Path, str]] = []

        for record in records:
            page_id = record["page_id"]
            text = record["text"].strip()
            if not text:
                continue
            crop_path = image_dir / f"{page_id}_crop.png"
            if crop_path.exists():
                self.samples.append((crop_path, text))
            else:
                print(f"  [skip] No crop found for {page_id}", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so it is ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Args & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on historical page crops.")
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=Path("data/ground_truth/ground_truth.jsonl"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/crops"),
        help="Directory with <page_id>_crop.png files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trocr_finetuned"))
    parser.add_argument(
        "--model-name",
        default="microsoft/trocr-base-printed",
        help="Base checkpoint to fine-tune from.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epoch count (for smoke tests).")
    parser.add_argument("--max-target-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {args.model_name}")

    records = read_jsonl(args.ground_truth_file)
    if not records:
        raise SystemExit("Ground truth file is empty.  Run parse_ground_truth.py first.")

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    # Required TrOCR config tweaks
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    dataset = TrOCRLineDataset(records, args.image_dir, processor, max_target_length=args.max_target_length)
    if not dataset.samples:
        raise SystemExit(
            f"No crop images found in {args.image_dir}.  Run run_baseline.py first to generate crops."
        )
    print(f"Fine-tuning on {len(dataset)} page crops.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        max_steps=args.max_steps,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"\nFine-tuned model saved to {output_dir}")
    print("To use in run_baseline.py: --model-name outputs/trocr_finetuned")


if __name__ == "__main__":
    main()
