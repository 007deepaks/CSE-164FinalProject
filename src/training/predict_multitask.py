"""Run multi-task test inference and create a Kaggle submission CSV."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import TestImageDataset
from src.training.classifier_utils import classifier_logits_with_tta, load_classifier_checkpoints
from src.training.multitask_utils import (
    load_multitask_checkpoint,
    semantic_mask_from_binary_and_class_logits,
    semantic_mask_from_logits,
)
from src.utils.masks import encode_mask_to_rle


@torch.no_grad()
def predict(
    checkpoint_path: Path,
    data_root: Path,
    output_csv: Path,
    image_size: int | None,
    batch_size: int,
    num_workers: int,
    max_test_samples: int | None,
    validate_with_starter: bool,
    tta: str,
    classifier_checkpoints: list[Path] | None,
    seg_threshold: float | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, saved_args = load_multitask_checkpoint(checkpoint_path, device)
    model.eval()
    classifier_models = load_classifier_checkpoints(classifier_checkpoints, device)
    resolved_image_size = image_size or int(saved_args.get("image_size", 320))
    dataset = TestImageDataset(data_root, resolved_image_size, max_test_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)
        classification_logits = outputs["classification"]
        if classifier_models:
            classification_logits = classifier_logits_with_tta(classifier_models, images, tta)
        if tta in {"hflip", "multi_crop"}:
            flipped_outputs = model(torch.flip(images, dims=(-1,)))
            flipped_classification_logits = flipped_outputs["classification"]
            if not classifier_models:
                classification_logits = 0.5 * (classification_logits + flipped_classification_logits)
            outputs = {
                "classification": classification_logits,
                "segmentation": 0.5
                * (outputs["segmentation"] + torch.flip(flipped_outputs["segmentation"], dims=(-1,))),
            }
        else:
            outputs["classification"] = classification_logits
        for item_index, image_name in enumerate(batch["image_name"]):
            height = int(batch["original_height"][item_index])
            width = int(batch["original_width"][item_index])
            if seg_threshold is None and not classifier_models:
                mask, class_id = semantic_mask_from_logits(outputs["segmentation"], outputs["classification"], item_index, height, width)
            else:
                mask, class_id = semantic_mask_from_binary_and_class_logits(
                    outputs["segmentation"],
                    outputs["classification"],
                    item_index,
                    height,
                    width,
                    seg_threshold,
                )
            rows.append(
                {
                    "image": str(image_name),
                    "class_id": class_id,
                    "segmentation_rle": encode_mask_to_rle(mask),
                }
            )
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  predicted batch {batch_index:04d}/{len(loader)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image", "class_id", "segmentation_rle"]).to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    if validate_with_starter and max_test_samples is None:
        subprocess.run(
            [
                sys.executable,
                "starter/validate_submission_csv.py",
                "--submission",
                str(output_csv),
                "--data-root",
                str(data_root),
                "--split",
                "test",
            ],
            check=True,
        )
    elif validate_with_starter:
        print("Skipping starter validation because --max-test-samples creates a partial submission.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_multitask.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submissions/submission.csv"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--tta", choices=["none", "hflip", "multi_crop"], default="none")
    parser.add_argument("--classifier-checkpoint", type=Path, nargs="+")
    parser.add_argument("--seg-threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_csv=args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_test_samples=args.max_test_samples,
        validate_with_starter=not args.no_validate,
        tta=args.tta,
        classifier_checkpoints=args.classifier_checkpoint,
        seg_threshold=args.seg_threshold,
    )


if __name__ == "__main__":
    main()
