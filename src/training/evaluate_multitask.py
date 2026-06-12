"""Evaluate a shared ConvNeXt multi-task checkpoint on validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import SegmentationDataset
from src.training.classifier_utils import load_classifier_checkpoints
from src.training.multitask_utils import binary_segmentation_bce_loss, load_multitask_checkpoint, validate_multitask
from src.utils.masks import IGNORE_ID


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    data_root: Path,
    image_size: int | None,
    batch_size: int,
    num_workers: int,
    max_val_samples: int | None,
    classifier_checkpoints: list[Path] | None,
    classifier_blend_weight: float,
    tta: str,
    seg_threshold: float | None,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, saved_args = load_multitask_checkpoint(checkpoint_path, device)
    classifier_models = load_classifier_checkpoints(classifier_checkpoints, device)
    resolved_image_size = image_size or int(saved_args.get("image_size", 320))
    dataset = SegmentationDataset(
        data_root,
        split="val",
        image_size=resolved_image_size,
        target_mode="binary",
        max_samples=max_val_samples,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    if int(saved_args.get("num_segmentation_classes", 2)) == 1:
        segmentation_criterion = binary_segmentation_bce_loss
    else:
        segmentation_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
    classification_criterion = nn.CrossEntropyLoss()
    return validate_multitask(
        model,
        loader,
        data_root,
        device,
        segmentation_criterion,
        classification_criterion,
        classifier_models=classifier_models,
        classifier_blend_weight=classifier_blend_weight,
        tta=tta,
        seg_threshold=seg_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_multitask.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--classifier-checkpoint", type=Path, nargs="+")
    parser.add_argument("--classifier-blend-weight", type=float, default=1.0)
    parser.add_argument("--tta", choices=["none", "hflip", "multi_crop"], default="none")
    parser.add_argument("--seg-threshold", type=float)
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
        classifier_checkpoints=args.classifier_checkpoint,
        classifier_blend_weight=args.classifier_blend_weight,
        tta=args.tta,
        seg_threshold=args.seg_threshold,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
