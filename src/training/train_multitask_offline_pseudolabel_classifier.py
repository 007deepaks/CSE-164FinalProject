"""Offline pseudo-label classifier-head tuning for a multi-task checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from src.data.segmentation_dataset import (
    augment_image_and_mask_to_tensors,
    augment_image_to_tensor,
    image_to_tensor,
    semantic_to_binary_mask,
)
from src.training.multitask_utils import (
    args_to_dict,
    binary_segmentation_bce_loss,
    load_multitask_checkpoint,
    validate_multitask,
)
from src.training.train_multitask import build_warmup_cosine_scheduler
from src.training.train_multitask_fixmatch import (
    ModelEma,
    autocast_settings,
    classifier_parameter_modules,
    freeze_for_classifier_only,
    forward_classification_chunks,
    keep_frozen_modules_eval,
    trainable_parameter_summary,
)
from src.utils.masks import IGNORE_ID, NUM_CLASSES, decode_rgb_mask


@dataclass(frozen=True)
class SupervisedSample:
    image_path: Path
    image_name: str
    class_id: int
    mask_path: Path | None = None


@dataclass(frozen=True)
class PseudoSample:
    image_path: Path
    image_name: str
    class_id: int
    confidence: float


class SupervisedClassifierHeadDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        data_root: Path,
        image_size: int,
        max_samples: int | None = None,
        augment: bool = True,
        random_crop: bool = True,
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.augment = augment
        self.random_crop = random_crop
        self.samples = self._load_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_samples(self) -> list[SupervisedSample]:
        labeled_rows = json.loads((self.data_root / "metadata" / "train_labeled.json").read_text(encoding="utf-8"))
        seg_rows = json.loads((self.data_root / "metadata" / "train_seg.json").read_text(encoding="utf-8"))
        samples = [
            SupervisedSample(
                image_path=self.data_root / str(row["image"]),
                image_name=Path(str(row["image"])).name,
                class_id=int(row["class_id"]),
            )
            for row in labeled_rows
        ]
        samples.extend(
            SupervisedSample(
                image_path=self.data_root / str(row["image"]),
                image_name=Path(str(row["image"])).name,
                class_id=int(row["class_id"]),
                mask_path=self.data_root / str(row["mask"]),
            )
            for row in seg_rows
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            if sample.mask_path is not None:
                semantic_mask_array = decode_rgb_mask(sample.mask_path).astype(np.int32)
                if self.augment:
                    image_tensor, semantic_mask = augment_image_and_mask_to_tensors(
                        image,
                        semantic_mask_array,
                        self.image_size,
                        self.random_crop,
                    )
                else:
                    image_tensor = image_to_tensor(image, self.image_size)
                    mask_image = Image.fromarray(semantic_mask_array, mode="I")
                    mask_image = mask_image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
                    semantic_mask = torch.from_numpy(np.asarray(mask_image, dtype=np.int64))
                mask = semantic_to_binary_mask(semantic_mask)
                has_mask = True
            else:
                image_tensor = (
                    augment_image_to_tensor(image, self.image_size, self.random_crop)
                    if self.augment
                    else image_to_tensor(image, self.image_size)
                )
                mask = torch.full((self.image_size, self.image_size), IGNORE_ID, dtype=torch.long)
                has_mask = False
        return {
            "image": image_tensor,
            "mask": mask,
            "has_mask": has_mask,
            "class_id": int(sample.class_id),
            "sample_weight": 1.0,
            "is_pseudo": False,
            "image_name": sample.image_name,
        }


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
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image_tensor = image_to_tensor(image.convert("RGB"), self.image_size)
        return {
            "image": image_tensor,
            "image_name": image_path.name,
            "image_path": str(image_path),
        }


class PseudoClassifierHeadDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        samples: list[PseudoSample],
        image_size: int,
        pseudo_weight: float,
        confidence_weight_power: float,
        augment: bool = True,
        random_crop: bool = True,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.pseudo_weight = pseudo_weight
        self.confidence_weight_power = confidence_weight_power
        self.augment = augment
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            image_tensor = (
                augment_image_to_tensor(image, self.image_size, self.random_crop)
                if self.augment
                else image_to_tensor(image, self.image_size)
            )
        sample_weight = self.pseudo_weight * (sample.confidence ** self.confidence_weight_power)
        return {
            "image": image_tensor,
            "mask": torch.full((self.image_size, self.image_size), IGNORE_ID, dtype=torch.long),
            "has_mask": False,
            "class_id": int(sample.class_id),
            "sample_weight": float(sample_weight),
            "is_pseudo": True,
            "image_name": sample.image_name,
        }


@torch.no_grad()
def mine_pseudo_labels(
    teacher: nn.Module,
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    confidence_threshold: float,
    max_per_class: int,
    max_unlabeled_samples: int | None,
    teacher_precision: str,
    tta: str,
    output_csv: Path | None,
) -> list[PseudoSample]:
    device = next(teacher.parameters()).device
    use_amp = device.type == "cuda"
    dataset = UnlabeledImageDataset(data_root, image_size, max_unlabeled_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    candidates_by_class: dict[int, list[PseudoSample]] = defaultdict(list)
    seen = 0
    finite = 0
    above_threshold = 0

    teacher.eval()
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        logits = forward_classification_chunks(
            teacher,
            images,
            use_amp,
            chunk_size=None,
            seg_forward=True,
            precision=teacher_precision,
        ).float()
        if tta == "hflip":
            flip_logits = forward_classification_chunks(
                teacher,
                torch.flip(images, dims=(-1,)),
                use_amp,
                chunk_size=None,
                seg_forward=True,
                precision=teacher_precision,
            ).float()
            logits = 0.5 * (logits + flip_logits)
        elif tta != "none":
            raise ValueError("tta must be 'none' or 'hflip'")
        finite_rows = torch.isfinite(logits).all(dim=1)
        probs = torch.softmax(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0), dim=1)
        confidences, class_ids = probs.max(dim=1)
        seen += int(logits.shape[0])
        finite += int(finite_rows.sum().item())
        for item_index, image_name in enumerate(batch["image_name"]):
            if not bool(finite_rows[item_index].item()):
                continue
            confidence = float(confidences[item_index].cpu().item())
            if confidence < confidence_threshold:
                continue
            above_threshold += 1
            class_id = int(class_ids[item_index].cpu().item())
            candidates_by_class[class_id].append(
                PseudoSample(
                    image_path=Path(str(batch["image_path"][item_index])),
                    image_name=str(image_name),
                    class_id=class_id,
                    confidence=confidence,
                )
            )
        if batch_index % 20 == 0 or batch_index == len(loader):
            pool_size = sum(len(items) for items in candidates_by_class.values())
            print(
                f"  mined batch {batch_index:04d}/{len(loader)} "
                f"seen={seen} finite={finite} accepted_pool={pool_size}"
            )

    selected: list[PseudoSample] = []
    for class_id, items in candidates_by_class.items():
        items.sort(key=lambda item: item.confidence, reverse=True)
        selected.extend(items[:max_per_class])
    selected.sort(key=lambda item: (item.class_id, -item.confidence, item.image_name))

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "class_id", "confidence"])
            writer.writeheader()
            for item in selected:
                writer.writerow(
                    {
                        "image": item.image_name,
                        "class_id": item.class_id,
                        "confidence": f"{item.confidence:.6f}",
                    }
                )
        print(f"Wrote pseudo labels: {output_csv}")

    counts = Counter(item.class_id for item in selected)
    mean_confidence = sum(item.confidence for item in selected) / max(1, len(selected))
    print(f"Seen unlabeled: {seen}")
    print(f"Finite teacher rows: {finite}")
    print(f"Above confidence threshold: {above_threshold}")
    print(f"Selected after class cap: {len(selected)}")
    print(f"Classes with >=1 pseudo-label: {len(counts)}")
    print(f"Mean selected confidence: {mean_confidence:.4f}")
    print("Top pseudo-label classes:")
    for class_id, count in counts.most_common(20):
        print(f"  class_id={class_id:03d} count={count}")
    return selected


def combined_sample_weights(dataset: ConcatDataset, supervised_weight: float, pseudo_sampler_weight: float) -> list[float]:
    weights: list[float] = []
    for sub_dataset in dataset.datasets:
        if isinstance(sub_dataset, SupervisedClassifierHeadDataset):
            weights.extend([supervised_weight] * len(sub_dataset))
        elif isinstance(sub_dataset, PseudoClassifierHeadDataset):
            weights.extend([pseudo_sampler_weight] * len(sub_dataset))
        else:
            weights.extend([1.0] * len(sub_dataset))
    return weights


def weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.cross_entropy(logits, targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    precision: str,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    keep_frozen_modules_eval(model)
    autocast_enabled, autocast_dtype = autocast_settings(use_amp, precision)
    total_loss = 0.0
    supervised_loss = 0.0
    pseudo_loss = 0.0
    total_batches = 0
    supervised_correct = 0
    supervised_total = 0
    pseudo_correct = 0
    pseudo_total = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        weights = batch["sample_weight"].to(device, non_blocking=True).float()
        is_pseudo = batch["is_pseudo"].to(device, non_blocking=True).bool()

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=autocast_enabled, dtype=autocast_dtype):
            outputs = model(images, seg=True)
            loss = weighted_cross_entropy(outputs["classification"], class_ids, weights)
        if not torch.isfinite(loss.detach()).all():
            print(f"  WARNING: skipping batch {batch_index:04d}; non-finite loss")
            continue
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(f"  WARNING: skipping batch {batch_index:04d}; non-finite grad_norm")
                continue
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            logits = outputs["classification"].detach()
            predictions = torch.argmax(logits, dim=1)
            total_loss += float(loss.item())
            if (~is_pseudo).any():
                supervised_mask = ~is_pseudo
                supervised_loss += float(
                    weighted_cross_entropy(logits[supervised_mask], class_ids[supervised_mask], weights[supervised_mask]).item()
                )
                supervised_correct += int((predictions[supervised_mask] == class_ids[supervised_mask]).sum().item())
                supervised_total += int(supervised_mask.sum().item())
            if is_pseudo.any():
                pseudo_loss += float(
                    weighted_cross_entropy(logits[is_pseudo], class_ids[is_pseudo], weights[is_pseudo]).item()
                )
                pseudo_correct += int((predictions[is_pseudo] == class_ids[is_pseudo]).sum().item())
                pseudo_total += int(is_pseudo.sum().item())
            total_batches += 1
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(f"  offline batch {batch_index:04d}/{len(loader)} loss={loss.item():.4f}")

    return {
        "train_loss": total_loss / max(1, total_batches),
        "supervised_loss": supervised_loss / max(1, total_batches),
        "pseudo_loss": pseudo_loss / max(1, total_batches),
        "supervised_accuracy": supervised_correct / max(1, supervised_total),
        "pseudo_accuracy": pseudo_correct / max(1, pseudo_total),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
    source_args: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_args = dict(source_args)
    saved_args.update(args_to_dict(args))
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_automated_score": best_score,
            "args": saved_args,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints/offline_pseudo_classifier"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--student-precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--teacher-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-per-class", type=int, default=25)
    parser.add_argument("--pseudo-weight", type=float, default=0.05)
    parser.add_argument("--pseudo-sampler-weight", type=float, default=0.35)
    parser.add_argument("--confidence-weight-power", type=float, default=1.0)
    parser.add_argument("--supervised-sampler-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-output", type=Path)
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    parser.add_argument("--max-unlabeled-samples", type=int)
    parser.add_argument("--max-supervised-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--validation-threshold", type=float, default=0.55)
    parser.add_argument("--no-random-crop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}; mixed precision: {use_amp}")
    model, _, source_args = load_multitask_checkpoint(args.resume_checkpoint, device)
    freeze_for_classifier_only(model)
    model.eval()
    trainable_count, total_count = trainable_parameter_summary(model)
    print(f"Loaded checkpoint: {args.resume_checkpoint}")
    print(f"Trainable parameters: {trainable_count:,}/{total_count:,}")

    pseudo_output = args.pseudo_output or (args.checkpoint_dir / "offline_pseudo_labels.csv")
    pseudo_samples = mine_pseudo_labels(
        teacher=model,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        confidence_threshold=args.confidence_threshold,
        max_per_class=args.max_per_class,
        max_unlabeled_samples=args.max_unlabeled_samples,
        teacher_precision=args.teacher_precision,
        tta=args.tta,
        output_csv=pseudo_output,
    )
    if not pseudo_samples:
        raise ValueError("No pseudo labels selected; lower --confidence-threshold or increase --max-per-class")

    supervised_dataset = SupervisedClassifierHeadDataset(
        args.data_root,
        image_size=args.image_size,
        max_samples=args.max_supervised_samples,
        augment=True,
        random_crop=not args.no_random_crop,
    )
    pseudo_dataset = PseudoClassifierHeadDataset(
        pseudo_samples,
        image_size=args.image_size,
        pseudo_weight=args.pseudo_weight,
        confidence_weight_power=args.confidence_weight_power,
        augment=True,
        random_crop=not args.no_random_crop,
    )
    train_dataset = ConcatDataset([supervised_dataset, pseudo_dataset])
    sampler = WeightedRandomSampler(
        combined_sample_weights(train_dataset, args.supervised_sampler_weight, args.pseudo_sampler_weight),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    from src.data.segmentation_dataset import SegmentationDataset

    val_dataset = SegmentationDataset(
        args.data_root,
        split="val",
        image_size=args.image_size,
        target_mode="binary",
        max_samples=args.max_val_samples,
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Train samples: supervised={len(supervised_dataset)} pseudo={len(pseudo_dataset)}; "
        f"val={len(val_dataset)}"
    )

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_learning_rate,
        base_lr=args.learning_rate,
    )
    if args.warmup_epochs > 0:
        first_epoch_lr = args.learning_rate / max(1, min(args.warmup_epochs, max(1, args.epochs - 1)))
        for group in optimizer.param_groups:
            group["lr"] = first_epoch_lr
    scaler = GradScaler(enabled=use_amp and args.student_precision == "fp16")
    validation_criterion = nn.CrossEntropyLoss()
    best_score = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            args.student_precision,
            args.gradient_clip,
        )
        val_metrics = validate_multitask(
            model,
            val_loader,
            args.data_root,
            device,
            binary_segmentation_bce_loss,
            validation_criterion,
            seg_threshold=args.validation_threshold,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "learning_rate": lr,
            "pseudo_count": len(pseudo_samples),
            "seg_threshold": args.validation_threshold,
        }
        history.append(row)
        print(
            "  "
            f"train_loss={row['train_loss']:.4f} "
            f"sup_loss={row['supervised_loss']:.4f} "
            f"pseudo_loss={row['pseudo_loss']:.4f} "
            f"sup_acc={row['supervised_accuracy']:.4f} "
            f"pseudo_acc={row['pseudo_accuracy']:.4f} "
            f"val_auto={row['automated_score']:.4f} "
            f"mIoU={row['mean_iou']:.4f} "
            f"bin_iou={row['binary_foreground_iou']:.4f} "
            f"oracle_mIoU={row['oracle_semantic_miou']:.4f} "
            f"rare_mIoU={row['rare_class_miou']:.4f} "
            f"macro_acc={row['classification_macro_accuracy']:.4f} "
            f"lr={lr:.6f}"
        )
        save_checkpoint(
            args.checkpoint_dir / "latest_multitask.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            args,
            source_args,
        )
        if row["automated_score"] > best_score:
            best_score = float(row["automated_score"])
            save_checkpoint(
                args.checkpoint_dir / "best_multitask.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                args,
                source_args,
            )
            print(f"  saved new best checkpoint with selection_metric={best_score:.4f}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.checkpoint_dir / "offline_pseudolabel_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nWrote history: {history_path}")


if __name__ == "__main__":
    main()
