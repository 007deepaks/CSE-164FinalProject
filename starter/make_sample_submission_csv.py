#!/usr/bin/env python3
"""Create an all-background CSV sample submission for Kaggle auto-scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--output", type=Path, default=Path("sample_submission.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.data_root / args.split / "images"
    images = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    rows = [{"image": image, "class_id": 0, "segmentation_rle": "0"} for image in images]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
