"""Save simple image, decoded-mask, and overlay visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.masks import IGNORE_ID, decode_rgb_mask, validate_mask_ids


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Create a deterministic RGB rendering for segmentation ids."""
    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    foreground = mask > 0
    colors[..., 0] = ((mask * 37) % 255).astype(np.uint8)
    colors[..., 1] = ((mask * 91) % 255).astype(np.uint8)
    colors[..., 2] = ((mask * 151) % 255).astype(np.uint8)
    colors[~foreground] = np.array([20, 20, 20], dtype=np.uint8)
    colors[mask == IGNORE_ID] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(colors, mode="RGB")


def make_panel(image_path: Path, mask_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    mask = decode_rgb_mask(mask_path)
    is_valid, invalid = validate_mask_ids(mask, allow_ignore=True)
    if not is_valid:
        print(f"WARNING: {mask_path} has invalid ids: {invalid.tolist()}")

    color_mask = colorize_mask(mask).resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.blend(image, color_mask, alpha=0.45)

    width, height = image.size
    panel = Image.new("RGB", (width * 3, height), color=(0, 0, 0))
    panel.paste(image, (0, 0))
    panel.paste(color_mask, (width, 0))
    panel.paste(overlay, (width * 2, 0))
    return panel


def load_metadata_rows(data_root: Path, split: str) -> list[dict[str, object]]:
    if split == "train_seg":
        return json.loads((data_root / "metadata" / "train_seg.json").read_text(encoding="utf-8"))
    if split == "val":
        images = sorted((data_root / "val" / "images").glob("*.JPEG"))
        return [
            {
                "image": f"val/images/{image_path.name}",
                "mask": f"val/masks/{image_path.with_suffix('.png').name}",
            }
            for image_path in images
        ]
    raise ValueError(f"Unsupported split for mask visualization: {split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", choices=["train_seg", "val"], default="train_seg")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_metadata_rows(args.data_root, args.split)
    for index, row in enumerate(rows[: args.num_samples]):
        image_path = args.data_root / str(row["image"])
        mask_path = args.data_root / str(row["mask"])
        panel = make_panel(image_path, mask_path)
        output_path = args.output_dir / f"{args.split}_mask_{index:03d}_{image_path.stem}.jpg"
        panel.save(output_path, quality=95)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
