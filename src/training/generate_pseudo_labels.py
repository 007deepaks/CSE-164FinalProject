"""Generate offline crop-aware pseudo-labels for train_unlabeled images."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.segmentation_dataset import image_to_tensor
from src.training.classifier_utils import (
    classifier_logits_full_and_seg_crop,
    classifier_logits_with_tta,
    load_classifier_checkpoints,
)
from src.training.multitask_utils import load_multitask_checkpoint


@dataclass(frozen=True)
class PseudoLabelCandidate:
    image: str
    class_id: int
    confidence: float


class UnlabeledImageDataset(Dataset[dict[str, object]]):
    def __init__(self, data_root: Path, image_size: int, max_samples: int | None = None) -> None:
        self.data_root = data_root
        self.image_size = image_size
        image_root = data_root / "train_unlabeled"
        paths = sorted(image_root.rglob("*.JPEG"))
        if not paths:
            paths = sorted(image_root.rglob("*.jpg")) + sorted(image_root.rglob("*.png"))
        if max_samples is not None:
            paths = paths[:max_samples]
        if not paths:
            raise ValueError(f"No unlabeled images found under {image_root}")
        self.image_paths = paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        from PIL import Image

        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            tensor = image_to_tensor(image.convert("RGB"), self.image_size)
        return {
            "image": tensor,
            "image_name": image_path.name,
        }


@torch.no_grad()
def generate_pseudo_labels(
    seg_checkpoint: Path,
    classifier_checkpoints: list[Path],
    data_root: Path,
    output_csv: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_unlabeled_samples: int | None,
    tta: str,
    seg_threshold: float,
    classifier_crop_mode: str,
    classifier_crop_padding: float,
    classifier_crop_weight: float,
    confidence_threshold: float,
    max_per_class: int,
    require_full_crop_agreement: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model, _, _ = load_multitask_checkpoint(seg_checkpoint, device)
    seg_model.eval()
    classifier_models = load_classifier_checkpoints(classifier_checkpoints, device)
    dataset = UnlabeledImageDataset(data_root, image_size, max_unlabeled_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    candidates_by_class: dict[int, list[PseudoLabelCandidate]] = defaultdict(list)
    seen = 0
    above_threshold = 0
    agreement_kept = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        outputs = seg_model(images)
        segmentation_logits = outputs["segmentation"]
        if tta in {"hflip", "multi_crop"}:
            flipped_outputs = seg_model(torch.flip(images, dims=(-1,)))
            segmentation_logits = 0.5 * (
                segmentation_logits + torch.flip(flipped_outputs["segmentation"], dims=(-1,))
            )

        full_logits = classifier_logits_with_tta(classifier_models, images, tta)
        if classifier_crop_mode == "seg":
            teacher_logits = classifier_logits_full_and_seg_crop(
                classifier_models,
                images,
                segmentation_logits,
                tta=tta,
                crop_threshold=seg_threshold,
                crop_padding=classifier_crop_padding,
                crop_weight=classifier_crop_weight,
            )
        else:
            teacher_logits = full_logits

        probabilities = torch.softmax(teacher_logits.float(), dim=1)
        confidences, class_ids = probabilities.max(dim=1)
        full_class_ids = torch.argmax(full_logits, dim=1)
        seen += len(images)
        for item_index, image_name in enumerate(batch["image_name"]):
            confidence = float(confidences[item_index].cpu().item())
            if confidence < confidence_threshold:
                continue
            above_threshold += 1
            class_id = int(class_ids[item_index].cpu().item())
            if require_full_crop_agreement and class_id != int(full_class_ids[item_index].cpu().item()):
                continue
            agreement_kept += 1
            candidates_by_class[class_id].append(
                PseudoLabelCandidate(
                    image=str(image_name),
                    class_id=class_id,
                    confidence=confidence,
                )
            )
        if batch_index % 20 == 0 or batch_index == len(loader):
            accepted_so_far = sum(len(values) for values in candidates_by_class.values())
            print(
                f"  pseudo-labeled batch {batch_index:04d}/{len(loader)} "
                f"seen={seen} accepted_pool={accepted_so_far}"
            )

    selected: list[PseudoLabelCandidate] = []
    for class_id, candidates in candidates_by_class.items():
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        selected.extend(candidates[:max_per_class])
    selected.sort(key=lambda item: (item.class_id, -item.confidence, item.image))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "class_id", "confidence"])
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "image": item.image,
                    "class_id": item.class_id,
                    "confidence": f"{item.confidence:.6f}",
                }
            )

    counts = Counter(item.class_id for item in selected)
    total_selected = len(selected)
    mean_confidence = sum(item.confidence for item in selected) / max(1, total_selected)
    print(f"Wrote {output_csv}")
    print(f"Seen unlabeled: {seen}")
    print(f"Above confidence threshold: {above_threshold}")
    print(f"After agreement filter: {agreement_kept}")
    print(f"Selected after class cap: {total_selected}")
    print(f"Classes with >=1 pseudo-label: {len(counts)}")
    print(f"Classes with >=25 pseudo-labels: {sum(1 for value in counts.values() if value >= 25)}")
    print(f"Classes with >=50 pseudo-labels: {sum(1 for value in counts.values() if value >= 50)}")
    print(f"Mean selected confidence: {mean_confidence:.4f}")
    print(f"Per-class selected min/mean/max: {min(counts.values(), default=0)}/"
          f"{(sum(counts.values()) / max(1, len(counts))):.1f}/{max(counts.values(), default=0)}")
    print("Top predicted classes:")
    for class_id, count in counts.most_common(20):
        print(f"  class_id={class_id:03d} count={count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pseudo_labels/pseudo_labels.csv"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-unlabeled-samples", type=int)
    parser.add_argument("--tta", choices=["none", "hflip", "multi_crop"], default="hflip")
    parser.add_argument("--seg-threshold", type=float, default=0.90)
    parser.add_argument("--classifier-crop-mode", choices=["none", "seg"], default="seg")
    parser.add_argument("--classifier-crop-padding", type=float, default=0.10)
    parser.add_argument("--classifier-crop-weight", type=float, default=0.50)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--max-per-class", type=int, default=50)
    parser.add_argument("--require-full-crop-agreement", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_pseudo_labels(
        seg_checkpoint=args.seg_checkpoint,
        classifier_checkpoints=args.classifier_checkpoint,
        data_root=args.data_root,
        output_csv=args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_unlabeled_samples=args.max_unlabeled_samples,
        tta=args.tta,
        seg_threshold=args.seg_threshold,
        classifier_crop_mode=args.classifier_crop_mode,
        classifier_crop_padding=args.classifier_crop_padding,
        classifier_crop_weight=args.classifier_crop_weight,
        confidence_threshold=args.confidence_threshold,
        max_per_class=args.max_per_class,
        require_full_crop_agreement=args.require_full_crop_agreement,
    )


if __name__ == "__main__":
    main()
