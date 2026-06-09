"""Tune segmentation-guided classifier crop settings on validation data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from starter.kaggle_metric import detailed_score
from src.data.segmentation_dataset import SegmentationDataset
from src.metrics.classification_metrics import ClassificationMetricTracker
from src.training.classifier_utils import (
    classifier_logits_with_tta,
    foreground_crop_tensors,
    load_classifier_checkpoints,
)
from src.training.multitask_utils import (
    build_val_solution_frame,
    load_multitask_checkpoint,
    semantic_mask_from_binary_and_class_logits,
)
from src.utils.masks import encode_mask_to_rle


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float")
    return values


@torch.no_grad()
def tune_crop_classifier(
    seg_checkpoint: Path,
    classifier_checkpoints: list[Path],
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_val_samples: int | None,
    tta: str,
    seg_thresholds: list[float],
    crop_paddings: list[float],
    crop_weights: list[float],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model, _, _ = load_multitask_checkpoint(seg_checkpoint, device)
    seg_model.eval()
    classifier_models = load_classifier_checkpoints(classifier_checkpoints, device)
    dataset = SegmentationDataset(
        data_root,
        split="val",
        image_size=image_size,
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
    grid = [
        (threshold, padding, weight)
        for threshold in seg_thresholds
        for padding in crop_paddings
        for weight in crop_weights
    ]
    rows_by_setting: dict[tuple[float, float, float], list[dict[str, object]]] = {setting: [] for setting in grid}
    class_trackers = {setting: ClassificationMetricTracker(num_classes=300) for setting in grid}

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        outputs = seg_model(images)
        segmentation_logits = outputs["segmentation"]
        if tta in {"hflip", "multi_crop"}:
            flipped_outputs = seg_model(torch.flip(images, dims=(-1,)))
            segmentation_logits = 0.5 * (
                segmentation_logits + torch.flip(flipped_outputs["segmentation"], dims=(-1,))
            )
        full_logits = classifier_logits_with_tta(classifier_models, images, tta)

        for threshold in seg_thresholds:
            for padding in crop_paddings:
                crop_images = foreground_crop_tensors(images, segmentation_logits, threshold, padding)
                crop_logits = classifier_logits_with_tta(classifier_models, crop_images, tta)
                for weight in crop_weights:
                    class_logits = (1.0 - weight) * full_logits + weight * crop_logits
                    class_trackers[(threshold, padding, weight)].update(class_logits, class_ids)
                    for item_index, image_name in enumerate(batch["image_name"]):
                        height = int(batch["original_height"][item_index])
                        width = int(batch["original_width"][item_index])
                        mask, class_id = semantic_mask_from_binary_and_class_logits(
                            segmentation_logits,
                            class_logits,
                            item_index,
                            height,
                            width,
                            threshold,
                        )
                        rows_by_setting[(threshold, padding, weight)].append(
                            {
                                "image": str(image_name),
                                "class_id": class_id,
                                "segmentation_rle": encode_mask_to_rle(mask),
                            }
                        )
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  tuned batch {batch_index:04d}/{len(loader)}")

    best_setting = None
    best_score = -1.0
    for setting in grid:
        threshold, padding, weight = setting
        submission = pd.DataFrame(rows_by_setting[setting], columns=["image", "class_id", "segmentation_rle"])
        metrics = detailed_score(solution, submission)
        class_metrics = class_trackers[setting].compute()
        print(
            f"seg_threshold={threshold:.3f} "
            f"crop_padding={padding:.3f} "
            f"crop_weight={weight:.3f} "
            f"automated={metrics['automated_score']:.4f} "
            f"seg={metrics['segmentation_score']:.4f} "
            f"mIoU={metrics['mean_iou']:.4f} "
            f"boundary={metrics['boundary_f_score']:.4f} "
            f"rare={metrics['rare_class_miou']:.4f} "
            f"macro_acc={metrics['classification_macro_accuracy']:.4f} "
            f"cls_top1={class_metrics['macro_accuracy']:.4f} "
            f"cls_top5={class_metrics['top5_accuracy']:.4f}"
        )
        if metrics["automated_score"] > best_score:
            best_score = float(metrics["automated_score"])
            best_setting = setting
    assert best_setting is not None
    print(
        "BEST "
        f"seg_threshold={best_setting[0]:.3f} "
        f"crop_padding={best_setting[1]:.3f} "
        f"crop_weight={best_setting[2]:.3f} "
        f"automated_score={best_score:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--tta", choices=["none", "hflip", "multi_crop"], default="hflip")
    parser.add_argument("--seg-thresholds", type=str, default="0.80,0.85,0.90,0.95")
    parser.add_argument("--crop-paddings", type=str, default="0.10,0.20,0.35,0.50")
    parser.add_argument("--crop-weights", type=str, default="0.30,0.50,0.70,0.90")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tune_crop_classifier(
        seg_checkpoint=args.seg_checkpoint,
        classifier_checkpoints=args.classifier_checkpoint,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_samples=args.max_val_samples,
        tta=args.tta,
        seg_thresholds=parse_float_list(args.seg_thresholds),
        crop_paddings=parse_float_list(args.crop_paddings),
        crop_weights=parse_float_list(args.crop_weights),
    )


if __name__ == "__main__":
    main()
