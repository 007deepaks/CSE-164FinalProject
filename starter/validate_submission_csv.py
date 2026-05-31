#!/usr/bin/env python3
"""Validate and optionally score a Kaggle CSV/RLE submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from kaggle_metric import NUM_CLASSES, decode_rle_to_mask, detailed_score, encode_mask_ids


def read_mask_ids(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint16)
    return arr[:, :, 0] + arr[:, :, 1] * 256


def build_solution_from_split(data_root: Path, split: str) -> pd.DataFrame:
    if split == "test":
        raise ValueError("Local test scoring requires the private Kaggle solution.csv")

    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"
    class_rows = json.loads((data_root / split / "classification.json").read_text())
    class_by_image = {row["image"]: int(row["class_id"]) for row in class_rows}
    rows = []
    for image_path in sorted(image_dir.glob("*.JPEG")):
        mask_path = mask_dir / image_path.with_suffix(".png").name
        with Image.open(image_path) as image:
            width, height = image.size
        rows.append(
            {
                "image": image_path.name,
                "height": height,
                "width": width,
                "class_id": class_by_image[image_path.name],
                "segmentation_rle": encode_mask_ids(read_mask_ids(mask_path)),
            }
        )
    return pd.DataFrame(rows)


def validate_submission_format(submission: pd.DataFrame, image_dir: Path) -> int:
    """Validate columns, image coverage, class ids, and RLE decodability."""
    required = {"image", "class_id", "segmentation_rle"}
    missing_columns = required - set(submission.columns)
    if missing_columns:
        raise ValueError(f"Submission is missing columns: {sorted(missing_columns)}")

    if submission["image"].duplicated().any():
        first = submission.loc[submission["image"].duplicated(), "image"].iloc[0]
        raise ValueError(f"Duplicate submission row for image {first}")

    expected_images = sorted(path.name for path in image_dir.glob("*.JPEG"))
    actual_images = set(submission["image"].astype(str))
    missing = sorted(set(expected_images) - actual_images)
    extra = sorted(actual_images - set(expected_images))
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} images; first: {missing[:5]}")
    if extra:
        raise ValueError(f"Extra predictions for {len(extra)} images; first: {extra[:5]}")

    result = submission.copy()
    result["image"] = result["image"].astype(str)
    try:
        result["class_id"] = result["class_id"].astype(int)
    except ValueError as exc:
        raise ValueError("class_id must be an integer") from exc

    bad = result[(result["class_id"] < 0) | (result["class_id"] >= NUM_CLASSES)]
    if not bad.empty:
        row = bad.iloc[0]
        raise ValueError(f"class_id out of range [0, {NUM_CLASSES - 1}] for {row['image']}")

    submission_by_image = result.set_index("image")
    for image_name in expected_images:
        with Image.open(image_dir / image_name) as image:
            width, height = image.size
        decode_rle_to_mask(submission_by_image.at[image_name, "segmentation_rle"], height, width)
    return len(expected_images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--solution-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = pd.read_csv(args.submission)
    image_dir = args.data_root / args.split / "images"
    if args.solution_csv:
        solution = pd.read_csv(args.solution_csv)
    elif args.split == "val":
        solution = build_solution_from_split(args.data_root, args.split)
    else:
        rows = validate_submission_format(submission, image_dir)
        print(json.dumps({"rows": rows, "scored": False, "status": "format ok"}, indent=2))
        return

    # Decode once for validation, then compute the full score if labels are available.
    solution_by_image = solution.set_index("image")
    submission_by_image = submission.set_index("image")
    for image_name, row in solution_by_image.iterrows():
        if image_name not in submission_by_image.index:
            raise ValueError(f"Missing row for {image_name}")
        with Image.open(image_dir / image_name) as image:
            width, height = image.size
        if int(row["height"]) != height or int(row["width"]) != width:
            raise ValueError(f"Solution metadata size mismatch for {image_name}")
        decode_rle_to_mask(submission_by_image.at[image_name, "segmentation_rle"], height, width)

    print(json.dumps(detailed_score(solution, submission), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
