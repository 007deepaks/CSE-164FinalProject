"""Run test inference and create a Kaggle submission CSV."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import TestImageDataset
from src.models.segmentation_model import build_segmentation_model
from src.utils.masks import encode_mask_to_rle, validate_prediction_mask


def infer_class_id(mask: np.ndarray) -> int:
    foreground = mask[mask > 0]
    if foreground.size == 0:
        return 0
    segmentation_id = Counter(foreground.astype(np.int64).tolist()).most_common(1)[0][0]
    return max(0, min(299, int(segmentation_id) - 1))


@torch.no_grad()
def predict(
    checkpoint_path: Path,
    data_root: Path,
    output_csv: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    base_channels: int | None,
    max_test_samples: int | None,
    validate_with_starter: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TestImageDataset(data_root, image_size, max_test_samples)
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

    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        for item_index, image_name in enumerate(batch["image_name"]):
            width = int(batch["original_width"][item_index])
            height = int(batch["original_height"][item_index])
            resized_logits = F.interpolate(
                logits[item_index : item_index + 1],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.argmax(resized_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint16)
            validate_prediction_mask(mask)
            rows.append(
                {
                    "image": str(image_name),
                    "class_id": infer_class_id(mask),
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
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_segmentation.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submissions/submission.csv"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--no-validate", action="store_true")
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
        base_channels=args.base_channels,
        max_test_samples=args.max_test_samples,
        validate_with_starter=not args.no_validate,
    )


if __name__ == "__main__":
    main()
