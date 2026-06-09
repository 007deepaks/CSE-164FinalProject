"""Visualize multi-task segmentation predictions against labeled validation masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import SegmentationDataset, TestImageDataset
from src.training.classifier_utils import classifier_logits_with_tta, load_classifier_checkpoints
from src.training.multitask_utils import load_multitask_checkpoint, semantic_mask_from_binary_and_class_logits
from src.utils.masks import IGNORE_ID, decode_rgb_mask


def colorize_mask(mask: np.ndarray) -> Image.Image:
    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colors[..., 0] = ((mask * 37) % 255).astype(np.uint8)
    colors[..., 1] = ((mask * 91) % 255).astype(np.uint8)
    colors[..., 2] = ((mask * 151) % 255).astype(np.uint8)
    colors[mask == 0] = np.array([20, 20, 20], dtype=np.uint8)
    colors[mask == IGNORE_ID] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(colors, mode="RGB")


def difference_panel(prediction: np.ndarray, ground_truth: np.ndarray) -> Image.Image:
    valid = ground_truth != IGNORE_ID
    diff = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    diff[(prediction == ground_truth) & (ground_truth > 0) & valid] = np.array([50, 210, 90], dtype=np.uint8)
    diff[(prediction > 0) & (ground_truth == 0) & valid] = np.array([230, 60, 60], dtype=np.uint8)
    diff[(prediction == 0) & (ground_truth > 0) & valid] = np.array([60, 120, 230], dtype=np.uint8)
    diff[(prediction != ground_truth) & (prediction > 0) & (ground_truth > 0) & valid] = np.array(
        [240, 190, 50],
        dtype=np.uint8,
    )
    diff[~valid] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(diff, mode="RGB")


def make_val_panel(data_root: Path, image_name: str, prediction: np.ndarray) -> Image.Image:
    image_path = data_root / "val" / "images" / image_name
    mask_path = data_root / "val" / "masks" / Path(image_name).with_suffix(".png").name
    image = Image.open(image_path).convert("RGB")
    ground_truth = decode_rgb_mask(mask_path)
    pred_color = colorize_mask(prediction)
    gt_color = colorize_mask(ground_truth)
    overlay = Image.blend(image, pred_color, alpha=0.45)
    diff = difference_panel(prediction, ground_truth)
    width, height = image.size
    panel = Image.new("RGB", (width * 5, height), color=(0, 0, 0))
    for offset, part in enumerate([image, gt_color, pred_color, overlay, diff]):
        panel.paste(part, (width * offset, 0))
    return panel


def make_test_panel(data_root: Path, image_name: str, prediction: np.ndarray) -> Image.Image:
    image_path = data_root / "test" / "images" / image_name
    image = Image.open(image_path).convert("RGB")
    pred_color = colorize_mask(prediction)
    overlay = Image.blend(image, pred_color, alpha=0.45)
    width, height = image.size
    panel = Image.new("RGB", (width * 3, height), color=(0, 0, 0))
    for offset, part in enumerate([image, pred_color, overlay]):
        panel.paste(part, (width * offset, 0))
    return panel


@torch.no_grad()
def visualize(
    checkpoint_path: Path,
    data_root: Path,
    split: str,
    image_size: int | None,
    num_samples: int,
    output_dir: Path,
    classifier_checkpoints: list[Path] | None,
    tta: str,
    seg_threshold: float | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, saved_args = load_multitask_checkpoint(checkpoint_path, device)
    model.eval()
    classifier_models = load_classifier_checkpoints(classifier_checkpoints, device)
    resolved_image_size = image_size or int(saved_args.get("image_size", 320))
    if split == "val":
        dataset = SegmentationDataset(
            data_root,
            split="val",
            image_size=resolved_image_size,
            target_mode="binary",
            max_samples=num_samples,
            augment=False,
        )
    else:
        dataset = TestImageDataset(data_root, image_size=resolved_image_size, max_samples=num_samples)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(loader):
        images = batch["image"].to(device)
        outputs = model(images)
        classification_logits = outputs["classification"]
        if classifier_models:
            classification_logits = classifier_logits_with_tta(classifier_models, images, tta)
        if tta in {"hflip", "multi_crop"}:
            flipped_images = torch.flip(images, dims=(-1,))
            flipped_outputs = model(flipped_images)
            flipped_classification_logits = flipped_outputs["classification"]
            if not classifier_models:
                classification_logits = 0.5 * (classification_logits + flipped_classification_logits)
            outputs = {
                "segmentation": 0.5
                * (outputs["segmentation"] + torch.flip(flipped_outputs["segmentation"], dims=(-1,))),
                "classification": classification_logits,
            }
        else:
            outputs["classification"] = classification_logits
        height = int(batch["original_height"][0])
        width = int(batch["original_width"][0])
        image_name = str(batch["image_name"][0])
        prediction, class_id = semantic_mask_from_binary_and_class_logits(
            outputs["segmentation"],
            outputs["classification"],
            0,
            height,
            width,
            seg_threshold,
        )
        panel = (
            make_val_panel(data_root, image_name, prediction)
            if split == "val"
            else make_test_panel(data_root, image_name, prediction)
        )
        output_path = output_dir / f"{split}_prediction_{index:03d}_{Path(image_name).stem}_class_{class_id:03d}.jpg"
        panel.save(output_path, quality=95)
        print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_multitask.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--classifier-checkpoint", type=Path, nargs="+")
    parser.add_argument("--tta", choices=["none", "hflip", "multi_crop"], default="none")
    parser.add_argument("--seg-threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        classifier_checkpoints=args.classifier_checkpoint,
        tta=args.tta,
        seg_threshold=args.seg_threshold,
    )


if __name__ == "__main__":
    main()
