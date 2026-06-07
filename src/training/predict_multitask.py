"""Run multi-task test inference and create a Kaggle submission CSV."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import TestImageDataset
from src.models.classification_model import build_classification_model, parse_depths
from src.training.multitask_utils import load_multitask_checkpoint, semantic_mask_from_logits
from src.utils.masks import NUM_CLASSES, encode_mask_to_rle, validate_prediction_mask


def load_classifier_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = build_classification_model(
        num_classes=NUM_CLASSES,
        base_channels=int(saved_args.get("base_channels", 48)),
        depths=parse_depths(str(saved_args.get("depths", "2,2,4,2"))),
        mlp_ratio=int(saved_args.get("mlp_ratio", 4)),
        drop_path=float(saved_args.get("drop_path", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def semantic_mask_from_binary_and_class_logits(
    segmentation_logits: torch.Tensor,
    classification_logits: torch.Tensor,
    index: int,
    height: int,
    width: int,
    seg_threshold: float | None,
) -> tuple[np.ndarray, int]:
    resized_logits = F.interpolate(
        segmentation_logits[index : index + 1],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    if seg_threshold is None:
        binary_prediction = torch.argmax(resized_logits, dim=1).squeeze(0).cpu().numpy()
    else:
        foreground_probability = torch.softmax(resized_logits, dim=1)[:, 1]
        binary_prediction = (foreground_probability.squeeze(0).cpu().numpy() > seg_threshold).astype(np.uint8)
    class_id = int(torch.argmax(classification_logits[index]).detach().cpu().item())
    mask = np.where(binary_prediction == 1, class_id + 1, 0).astype(np.uint16)
    validate_prediction_mask(mask)
    return mask, class_id


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
    classifier_checkpoint: Path | None,
    seg_threshold: float | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, saved_args = load_multitask_checkpoint(checkpoint_path, device)
    model.eval()
    classifier_model = load_classifier_checkpoint(classifier_checkpoint, device) if classifier_checkpoint else None
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
        if classifier_model is not None:
            classification_logits = classifier_model(images)
        if tta == "hflip":
            flipped_outputs = model(torch.flip(images, dims=(-1,)))
            flipped_classification_logits = flipped_outputs["classification"]
            if classifier_model is not None:
                flipped_classification_logits = classifier_model(torch.flip(images, dims=(-1,)))
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
            if seg_threshold is None and classifier_model is None:
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
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument("--classifier-checkpoint", type=Path)
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
        classifier_checkpoint=args.classifier_checkpoint,
        seg_threshold=args.seg_threshold,
    )


if __name__ == "__main__":
    main()
