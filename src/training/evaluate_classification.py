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
from src.models.classification_model import build_classification_model, parse_depths


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_val_samples: int | None,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    base_channels = int(saved_args.get("base_channels", 48))
    depths = parse_depths(str(saved_args.get("depths", "2,2,4,2")))
    mlp_ratio = int(saved_args.get("mlp_ratio", 4))
    drop_path = float(saved_args.get("drop_path", 0.0))

    dataset = ClassificationDataset(data_root, "val", image_size, max_val_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_classification_model(
        num_classes=300,
        base_channels=base_channels,
        depths=depths,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    tracker = ClassificationMetricTracker(num_classes=300)
    running_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["class_id"].to(device, non_blocking=True)
        logits = model(images)
        running_loss += float(criterion(logits, targets).item())
        tracker.update(logits, targets)
    metrics = tracker.compute()
    metrics["loss"] = running_loss / max(1, len(loader))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_classification.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_samples=args.max_val_samples,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
