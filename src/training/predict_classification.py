"""Predict class_id values for test images and write image,class_id CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.classification_dataset import ClassificationDataset
from src.models.classification_model import build_classification_model, parse_depths


@torch.no_grad()
def predict(
    checkpoint_path: Path,
    data_root: Path,
    output_csv: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_test_samples: int | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    base_channels = int(saved_args.get("base_channels", 48))
    depths = parse_depths(str(saved_args.get("depths", "2,2,4,2")))
    mlp_ratio = int(saved_args.get("mlp_ratio", 4))
    drop_path = float(saved_args.get("drop_path", 0.0))

    dataset = ClassificationDataset(data_root, "test", image_size, max_test_samples)
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

    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        predictions = torch.argmax(logits, dim=1).cpu().tolist()
        for image_name, class_id in zip(batch["image_name"], predictions):
            rows.append({"image": str(image_name), "class_id": int(class_id)})
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  predicted batch {batch_index:04d}/{len(loader)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image", "class_id"]).to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_classification.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions/test_class_predictions.csv"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int)
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
    )


if __name__ == "__main__":
    main()
