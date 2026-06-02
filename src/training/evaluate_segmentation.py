"""Evaluate a trained segmentation checkpoint on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import SegmentationDataset
from src.metrics.segmentation_metrics import SegmentationMetricTracker
from src.models.segmentation_model import build_segmentation_model
from src.training.train_segmentation import save_prediction_panels
from src.utils.masks import IGNORE_ID


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    base_channels: int | None,
    max_val_samples: int | None,
    figure_dir: Path,
    num_visualizations: int,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(data_root, "val", image_size, max_val_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_base_channels = base_channels or int(checkpoint.get("args", {}).get("base_channels", 32))
    model = build_segmentation_model(num_classes=301, base_channels=resolved_base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
    tracker = SegmentationMetricTracker(ignore_index=IGNORE_ID)
    running_loss = 0.0
    saved_visuals = False

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        running_loss += float(criterion(logits, masks).item())
        tracker.update(logits, masks)
        if not saved_visuals and num_visualizations > 0:
            save_prediction_panels(
                images.cpu(),
                masks.cpu(),
                logits.cpu(),
                figure_dir,
                prefix="eval_val",
                max_examples=num_visualizations,
            )
            saved_visuals = True

    metrics = tracker.compute()
    metrics["loss"] = running_loss / max(1, len(loader))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_segmentation.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--num-visualizations", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        max_val_samples=args.max_val_samples,
        figure_dir=args.figure_dir,
        num_visualizations=args.num_visualizations,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
