"""Evaluate a classification checkpoint on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.classification_dataset import ClassificationDataset
from src.metrics.classification_metrics import ClassificationMetricTracker
from src.training.classifier_utils import classifier_logits_with_tta, load_classifier_checkpoints


@torch.no_grad()
def evaluate(
    checkpoint_paths: list[Path],
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_val_samples: int | None,
    tta: str,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ClassificationDataset(data_root, "val", image_size, max_val_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    models = load_classifier_checkpoints(checkpoint_paths, device)

    criterion = nn.CrossEntropyLoss()
    tracker = ClassificationMetricTracker(num_classes=300)
    running_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["class_id"].to(device, non_blocking=True)
        logits = classifier_logits_with_tta(models, images, tta)
        running_loss += float(criterion(logits, targets).item())
        tracker.update(logits, targets)
    metrics = tracker.compute()
    metrics["loss"] = running_loss / max(1, len(loader))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        default=[Path("outputs/checkpoints/best_classification.pt")],
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--tta", choices=["none", "hflip", "multi_crop"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        checkpoint_paths=args.checkpoint,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_samples=args.max_val_samples,
        tta=args.tta,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
