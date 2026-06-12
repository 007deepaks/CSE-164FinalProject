"""Tune foreground probability threshold for multi-task segmentation submissions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from starter.kaggle_metric import detailed_score
from src.data.segmentation_dataset import SegmentationDataset
from src.training.classifier_utils import classifier_logits_with_tta, load_classifier_checkpoints
from src.training.multitask_utils import (
    blend_classification_logits,
    build_val_solution_frame,
    load_multitask_checkpoint,
    semantic_mask_from_binary_and_class_logits,
)
from src.utils.masks import encode_mask_to_rle


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


@torch.no_grad()
def score_thresholds(
    seg_checkpoint: Path,
    classifier_checkpoints: list[Path] | None,
    data_root: Path,
    image_size: int | None,
    batch_size: int,
    num_workers: int,
    max_val_samples: int | None,
    thresholds: list[float],
    tta: str,
    classifier_blend_weight: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model, _, saved_args = load_multitask_checkpoint(seg_checkpoint, device)
    seg_model.eval()
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
    solution = build_val_solution_frame(data_root, {sample.image_name for sample in dataset.samples})
    rows_by_threshold: dict[float, list[dict[str, object]]] = {threshold: [] for threshold in thresholds}

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        outputs = seg_model(images)
        classification_logits = outputs["classification"]
        if tta in {"hflip", "multi_crop"}:
            flipped_images = torch.flip(images, dims=(-1,))
            flipped_outputs = seg_model(flipped_images)
            flipped_classification_logits = flipped_outputs["classification"]
            classification_logits = 0.5 * (classification_logits + flipped_classification_logits)
            outputs = {
                "segmentation": 0.5
                * (outputs["segmentation"] + torch.flip(flipped_outputs["segmentation"], dims=(-1,))),
                "classification": classification_logits,
            }
        else:
            outputs["classification"] = classification_logits
        if classifier_models:
            external_classification_logits = classifier_logits_with_tta(classifier_models, images, tta)
            outputs["classification"] = blend_classification_logits(
                outputs["classification"],
                external_classification_logits,
                classifier_blend_weight,
            )

        for item_index, image_name in enumerate(batch["image_name"]):
            height = int(batch["original_height"][item_index])
            width = int(batch["original_width"][item_index])
            for threshold in thresholds:
                mask, class_id = semantic_mask_from_binary_and_class_logits(
                    outputs["segmentation"],
                    outputs["classification"],
                    item_index,
                    height,
                    width,
                    threshold,
                )
                rows_by_threshold[threshold].append(
                    {
                        "image": str(image_name),
                        "class_id": class_id,
                        "segmentation_rle": encode_mask_to_rle(mask),
                    }
                )
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  scored batch {batch_index:04d}/{len(loader)}")

    best_threshold = None
    best_score = -1.0
    for threshold in thresholds:
        submission = pd.DataFrame(rows_by_threshold[threshold], columns=["image", "class_id", "segmentation_rle"])
        metrics = detailed_score(solution, submission)
        print(
            f"threshold={threshold:.3f} "
            f"automated={metrics['automated_score']:.4f} "
            f"seg={metrics['segmentation_score']:.4f} "
            f"mIoU={metrics['mean_iou']:.4f} "
            f"boundary={metrics['boundary_f_score']:.4f} "
            f"rare={metrics['rare_class_miou']:.4f} "
            f"macro_acc={metrics['classification_macro_accuracy']:.4f}"
        )
        if metrics["automated_score"] > best_score:
            best_score = float(metrics["automated_score"])
            best_threshold = threshold
    print(f"BEST threshold={best_threshold:.3f} automated_score={best_score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--thresholds", type=str, default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--tta", choices=["none", "hflip", "multi_crop"], default="none")
    parser.add_argument("--classifier-blend-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_thresholds(
        seg_checkpoint=args.seg_checkpoint,
        classifier_checkpoints=args.classifier_checkpoint,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_samples=args.max_val_samples,
        thresholds=parse_thresholds(args.thresholds),
        tta=args.tta,
        classifier_blend_weight=args.classifier_blend_weight,
    )


if __name__ == "__main__":
    main()
