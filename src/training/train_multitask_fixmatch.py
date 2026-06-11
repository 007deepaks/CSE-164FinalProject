"""Online FixMatch-style SSL for the multi-task segmentation/classification model."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.classification_dataset import UnlabeledFixMatchDataset
from src.data.segmentation_dataset import (
    augment_image_and_mask_to_tensors,
    augment_image_to_tensor,
    image_to_tensor,
    semantic_to_binary_mask,
)
from src.training.multitask_utils import (
    args_to_dict,
    binary_segmentation_bce_loss,
    binary_segmentation_dice_loss,
    load_multitask_checkpoint,
    validate_multitask,
)
from src.training.train_multitask import build_warmup_cosine_scheduler
from src.utils.masks import IGNORE_ID, NUM_CLASSES, decode_rgb_mask


@dataclass(frozen=True)
class SupervisedMultiTaskSample:
    image_path: Path
    image_name: str
    class_id: int
    mask_path: Path | None = None


class SupervisedMultiTaskDataset(Dataset[dict[str, object]]):
    """Combined train_labeled + train_seg dataset with optional masks."""

    def __init__(
        self,
        data_root: str | Path,
        image_size: int,
        max_samples: int | None = None,
        augment: bool = True,
        random_crop: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        self.random_crop = random_crop
        self.samples = self._load_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_samples(self) -> list[SupervisedMultiTaskSample]:
        labeled_rows = json.loads((self.data_root / "metadata" / "train_labeled.json").read_text(encoding="utf-8"))
        seg_rows = json.loads((self.data_root / "metadata" / "train_seg.json").read_text(encoding="utf-8"))
        samples = [
            SupervisedMultiTaskSample(
                image_path=self.data_root / str(row["image"]),
                image_name=Path(str(row["image"])).name,
                class_id=int(row["class_id"]),
            )
            for row in labeled_rows
        ]
        samples.extend(
            SupervisedMultiTaskSample(
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
            "image_name": sample.image_name,
        }


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.num_updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


class DistributionAlignment:
    def __init__(self, num_classes: int = NUM_CLASSES, momentum: float = 0.999) -> None:
        self.num_classes = num_classes
        self.momentum = momentum
        self.p_model: torch.Tensor | None = None

    @torch.no_grad()
    def adjust(self, probs: torch.Tensor) -> torch.Tensor:
        batch_mean = probs.detach().mean(dim=0)
        if self.p_model is None:
            self.p_model = torch.full_like(batch_mean, 1.0 / self.num_classes)
        self.p_model.mul_(self.momentum).add_(batch_mean, alpha=1.0 - self.momentum)
        adjusted = probs * ((1.0 / self.num_classes) / (self.p_model.to(probs.device) + 1e-6))
        return adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(1e-6)


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def supervised_sample_weights(samples: list[SupervisedMultiTaskSample], class_weight: float, mask_weight: float) -> list[float]:
    return [mask_weight if sample.mask_path is not None else class_weight for sample in samples]


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


def forward_classification_chunks(
    model: nn.Module,
    images: torch.Tensor,
    use_amp: bool,
    chunk_size: int | None,
    seg_forward: bool,
) -> torch.Tensor:
    chunks = images.split(chunk_size or len(images))
    logits: list[torch.Tensor] = []
    for chunk in chunks:
        with autocast(device_type="cuda", enabled=use_amp):
            logits.append(model(chunk, seg=seg_forward)["classification"])
    return torch.cat(logits, dim=0)


def train_one_epoch(
    model: nn.Module,
    ema: ModelEma,
    supervised_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    classification_criterion: nn.Module,
    args: argparse.Namespace,
    da: DistributionAlignment | None,
) -> dict[str, float]:
    model.train()
    ema.module.eval()
    supervised_iter = infinite_loader(supervised_loader)
    unlabeled_iter = infinite_loader(unlabeled_loader)
    steps = args.steps_per_epoch or max(len(supervised_loader), int(len(unlabeled_loader) / max(1.0, args.unlabeled_ratio)))

    totals = {
        "supervised_cls_loss": 0.0,
        "supervised_seg_loss": 0.0,
        "supervised_dice_loss": 0.0,
        "unsupervised_cls_loss": 0.0,
        "total_loss": 0.0,
        "accepted": 0.0,
        "seen": 0.0,
        "confidence_sum": 0.0,
        "raw_confidence_sum": 0.0,
        "adjusted_confidence_sum": 0.0,
        "raw_confidence_max": 0.0,
        "adjusted_confidence_max": 0.0,
        "raw_accept": 0.0,
        "steps": 0.0,
    }
    class_hist = torch.zeros(NUM_CLASSES, dtype=torch.long)
    supervised_correct = 0
    supervised_total = 0

    for step in range(1, steps + 1):
        supervised_batch = next(supervised_iter)
        unlabeled_batch = next(unlabeled_iter)

        images = supervised_batch["image"].to(device, non_blocking=True)
        masks = supervised_batch["mask"].to(device, non_blocking=True)
        has_mask = supervised_batch["has_mask"].to(device, non_blocking=True).bool()
        class_ids = supervised_batch["class_id"].to(device, non_blocking=True)
        weak_images = unlabeled_batch["weak_image"].to(device, non_blocking=True)
        strong_images = unlabeled_batch["strong_image"].to(device, non_blocking=True)

        with torch.no_grad():
            teacher_logits = forward_classification_chunks(
                ema.module,
                weak_images,
                use_amp,
                args.weak_forward_batch_size,
                seg_forward=not args.ssl_fast_classifier,
            ).float()
            raw_probs = torch.softmax(teacher_logits, dim=1)
            raw_confidence, _ = raw_probs.max(dim=1)
            probs = raw_probs
            if da is not None:
                probs = da.adjust(probs)
            confidence, pseudo_targets = probs.max(dim=1)
            accepted_mask = confidence.ge(args.confidence_threshold)
            raw_accept_mask = raw_confidence.ge(args.confidence_threshold)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            supervised_outputs = model(images, seg=True)
            supervised_cls_loss = classification_criterion(supervised_outputs["classification"], class_ids)
            if has_mask.any():
                seg_logits = supervised_outputs["segmentation"][has_mask]
                seg_masks = masks[has_mask]
                supervised_bce = binary_segmentation_bce_loss(seg_logits, seg_masks)
                supervised_dice = binary_segmentation_dice_loss(seg_logits.float(), seg_masks)
                supervised_seg_loss = args.segmentation_loss_weight * supervised_bce + args.dice_loss_weight * supervised_dice
            else:
                supervised_bce = supervised_outputs["classification"].sum() * 0.0
                supervised_dice = supervised_outputs["classification"].sum() * 0.0
                supervised_seg_loss = supervised_outputs["classification"].sum() * 0.0

            if accepted_mask.any():
                strong_logits = forward_classification_chunks(
                    model,
                    strong_images[accepted_mask],
                    use_amp,
                    args.strong_forward_batch_size,
                    seg_forward=not args.ssl_fast_classifier,
                )
                unsupervised_cls_loss = nn.functional.cross_entropy(strong_logits, pseudo_targets[accepted_mask])
            else:
                unsupervised_cls_loss = supervised_outputs["classification"].sum() * 0.0

            loss = supervised_cls_loss + supervised_seg_loss + args.unlabeled_loss_weight * unsupervised_cls_loss

        components = {
            "supervised_cls_loss": supervised_cls_loss,
            "supervised_seg_loss": supervised_seg_loss,
            "unsupervised_cls_loss": unsupervised_cls_loss,
            "loss": loss,
        }
        bad = [name for name, value in components.items() if not torch.isfinite(value.detach()).all()]
        if bad:
            print(f"  WARNING: skipping ssl step {step:04d}; non-finite {bad}")
            continue

        scaler.scale(loss).backward()
        if args.gradient_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(f"  WARNING: skipping ssl step {step:04d}; non-finite grad_norm")
                continue
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        with torch.no_grad():
            predictions = torch.argmax(supervised_outputs["classification"], dim=1)
            supervised_correct += int((predictions == class_ids).sum().item())
            supervised_total += int(class_ids.numel())
            accepted_count = int(accepted_mask.sum().item())
            if accepted_count:
                accepted_targets_cpu = pseudo_targets[accepted_mask].detach().cpu()
                class_hist += torch.bincount(accepted_targets_cpu, minlength=NUM_CLASSES)
                totals["confidence_sum"] += float(confidence[accepted_mask].sum().item())
            totals["accepted"] += float(accepted_count)
            totals["seen"] += float(accepted_mask.numel())
            totals["raw_accept"] += float(raw_accept_mask.sum().item())
            totals["raw_confidence_sum"] += float(raw_confidence.sum().item())
            totals["adjusted_confidence_sum"] += float(confidence.sum().item())
            totals["raw_confidence_max"] = max(totals["raw_confidence_max"], float(raw_confidence.max().item()))
            totals["adjusted_confidence_max"] = max(totals["adjusted_confidence_max"], float(confidence.max().item()))
            totals["supervised_cls_loss"] += float(supervised_cls_loss.item())
            totals["supervised_seg_loss"] += float(supervised_bce.item())
            totals["supervised_dice_loss"] += float(supervised_dice.item())
            totals["unsupervised_cls_loss"] += float(unsupervised_cls_loss.item())
            totals["total_loss"] += float(loss.item())
            totals["steps"] += 1.0

        if step % args.print_every == 0 or step == steps:
            accepted_total = max(1.0, totals["accepted"])
            print(
                f"  ssl step {step:04d}/{steps} "
                f"sup_cls={supervised_cls_loss.item():.4f} "
                f"sup_seg={supervised_seg_loss.item():.4f} "
                f"unsup={unsupervised_cls_loss.item():.4f} "
                f"accept={totals['accepted'] / max(1.0, totals['seen']):.3f} "
                f"raw_accept={totals['raw_accept'] / max(1.0, totals['seen']):.3f} "
                f"raw_conf={totals['raw_confidence_sum'] / max(1.0, totals['seen']):.3f}/{totals['raw_confidence_max']:.3f} "
                f"adj_conf={totals['adjusted_confidence_sum'] / max(1.0, totals['seen']):.3f}/{totals['adjusted_confidence_max']:.3f} "
                f"accepted_conf={totals['confidence_sum'] / accepted_total:.3f}"
            )

    divisor = max(1.0, totals["steps"])
    covered_classes = int((class_hist > 0).sum().item())
    top_counts, top_classes = torch.topk(class_hist, k=min(10, NUM_CLASSES))
    top_hist = ",".join(f"{int(cls)}:{int(count)}" for cls, count in zip(top_classes, top_counts) if int(count) > 0)
    return {
        "supervised_cls_loss": totals["supervised_cls_loss"] / divisor,
        "supervised_seg_loss": totals["supervised_seg_loss"] / divisor,
        "supervised_dice_loss": totals["supervised_dice_loss"] / divisor,
        "unsupervised_cls_loss": totals["unsupervised_cls_loss"] / divisor,
        "total_loss": totals["total_loss"] / divisor,
        "pseudo_acceptance_rate": totals["accepted"] / max(1.0, totals["seen"]),
        "pseudo_raw_acceptance_rate": totals["raw_accept"] / max(1.0, totals["seen"]),
        "pseudo_mean_confidence": totals["confidence_sum"] / max(1.0, totals["accepted"]),
        "pseudo_raw_mean_confidence": totals["raw_confidence_sum"] / max(1.0, totals["seen"]),
        "pseudo_adjusted_mean_confidence": totals["adjusted_confidence_sum"] / max(1.0, totals["seen"]),
        "pseudo_raw_max_confidence": totals["raw_confidence_max"],
        "pseudo_adjusted_max_confidence": totals["adjusted_confidence_max"],
        "pseudo_class_coverage": float(covered_classes),
        "supervised_train_accuracy": supervised_correct / max(1, supervised_total),
        "pseudo_top_hist": top_hist,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints/multitask_ssl"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unlabeled-batch-size", type=int, default=32)
    parser.add_argument("--unlabeled-ratio", type=float, default=2.0)
    parser.add_argument("--weak-forward-batch-size", type=int)
    parser.add_argument("--strong-forward-batch-size", type=int)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--segmentation-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--unlabeled-loss-weight", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--distribution-alignment", action="store_true")
    parser.add_argument("--distribution-alignment-momentum", type=float, default=0.999)
    parser.add_argument("--class-only-sample-weight", type=float, default=1.0)
    parser.add_argument("--mask-sample-weight", type=float, default=2.5)
    parser.add_argument("--max-supervised-samples", type=int)
    parser.add_argument("--max-unlabeled-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--diagnose-teacher-only", action="store_true")
    parser.add_argument("--diagnose-batches", type=int, default=20)
    parser.add_argument("--validation-threshold", type=float, default=0.55)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--full-val-every", type=int, default=1)
    parser.add_argument("--quick-val-samples", type=int)
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--unlabeled-random-crop", action="store_true")
    parser.add_argument(
        "--ssl-fast-classifier",
        action="store_true",
        help="Use seg=False for unlabeled weak/strong forwards. Faster, but not mask-guided.",
    )
    return parser.parse_args()


@torch.no_grad()
def diagnose_teacher_confidence(
    teacher: nn.Module,
    unlabeled_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
    da: DistributionAlignment | None,
) -> None:
    teacher.eval()
    raw_confidences: list[torch.Tensor] = []
    adjusted_confidences: list[torch.Tensor] = []
    raw_classes = torch.zeros(NUM_CLASSES, dtype=torch.long)
    adjusted_classes = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for batch_index, batch in enumerate(unlabeled_loader, start=1):
        weak_images = batch["weak_image"].to(device, non_blocking=True)
        logits = forward_classification_chunks(
            teacher,
            weak_images,
            use_amp,
            args.weak_forward_batch_size,
            seg_forward=not args.ssl_fast_classifier,
        ).float()
        raw_probs = torch.softmax(logits, dim=1)
        raw_conf, raw_target = raw_probs.max(dim=1)
        adjusted_probs = da.adjust(raw_probs) if da is not None else raw_probs
        adjusted_conf, adjusted_target = adjusted_probs.max(dim=1)
        raw_confidences.append(raw_conf.cpu())
        adjusted_confidences.append(adjusted_conf.cpu())
        raw_classes += torch.bincount(raw_target.cpu(), minlength=NUM_CLASSES)
        adjusted_classes += torch.bincount(adjusted_target.cpu(), minlength=NUM_CLASSES)
        if batch_index >= args.diagnose_batches:
            break

    raw = torch.cat(raw_confidences)
    adjusted = torch.cat(adjusted_confidences)
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print("\nTeacher confidence diagnosis")
    print(f"  samples={raw.numel()} DA={da is not None} ssl_seg_forward={not args.ssl_fast_classifier}")
    for name, values, classes in [("raw", raw, raw_classes), ("adjusted", adjusted, adjusted_classes)]:
        quantiles = torch.quantile(values, torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99]))
        accept_parts = [f"{threshold:.2f}:{float((values >= threshold).float().mean().item()):.3f}" for threshold in thresholds]
        top_counts, top_classes = torch.topk(classes, k=10)
        top_hist = ",".join(
            f"{int(cls)}:{int(count)}" for cls, count in zip(top_classes, top_counts) if int(count) > 0
        )
        print(
            f"  {name}: mean={float(values.mean()):.4f} max={float(values.max()):.4f} "
            f"p50={float(quantiles[0]):.4f} p75={float(quantiles[1]):.4f} "
            f"p90={float(quantiles[2]):.4f} p95={float(quantiles[3]):.4f} "
            f"p99={float(quantiles[4]):.4f}"
        )
        print(f"  {name} acceptance: {' '.join(accept_parts)}")
        print(f"  {name} top_classes: {top_hist}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}; mixed precision: {use_amp}")
    print(f"Loading supervised checkpoint: {args.resume_checkpoint}")
    model, _, source_args = load_multitask_checkpoint(args.resume_checkpoint, device)
    model.train()
    ema = ModelEma(model, args.ema_decay)

    supervised_dataset = SupervisedMultiTaskDataset(
        args.data_root,
        image_size=args.image_size,
        max_samples=args.max_supervised_samples,
        augment=True,
        random_crop=not args.no_random_crop,
    )
    supervised_sampler = WeightedRandomSampler(
        supervised_sample_weights(supervised_dataset.samples, args.class_only_sample_weight, args.mask_sample_weight),
        num_samples=len(supervised_dataset),
        replacement=True,
    )
    supervised_loader = DataLoader(
        supervised_dataset,
        batch_size=args.batch_size,
        sampler=supervised_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    unlabeled_dataset = UnlabeledFixMatchDataset(
        args.data_root,
        image_size=args.image_size,
        max_samples=args.max_unlabeled_samples,
        random_crop=args.unlabeled_random_crop,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.unlabeled_batch_size,
        shuffle=True,
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
    quick_val_loader = None
    if args.quick_val_samples is not None and args.quick_val_samples < len(val_dataset):
        quick_val_dataset = SegmentationDataset(
            args.data_root,
            split="val",
            image_size=args.image_size,
            target_mode="binary",
            max_samples=args.quick_val_samples,
            augment=False,
        )
        quick_val_loader = DataLoader(
            quick_val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    print(
        f"Supervised samples: {len(supervised_dataset)}; "
        f"unlabeled samples: {len(unlabeled_dataset)}; val samples: {len(val_dataset)}"
    )
    print(
        f"SSL: threshold={args.confidence_threshold}, unlabeled_weight={args.unlabeled_loss_weight}, "
        f"DA={args.distribution_alignment}, ssl_seg_forward={not args.ssl_fast_classifier}"
    )

    classification_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    validation_classification_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
    scaler = GradScaler(enabled=use_amp)
    da = DistributionAlignment(NUM_CLASSES, args.distribution_alignment_momentum) if args.distribution_alignment else None
    if args.diagnose_teacher_only:
        diagnose_teacher_confidence(ema.module, unlabeled_loader, device, use_amp, args, da)
        return
    best_score = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model,
            ema,
            supervised_loader,
            unlabeled_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            classification_criterion,
            args,
            da,
        )
        should_validate = epoch % max(1, args.validate_every) == 0 or epoch == args.epochs
        should_full_validate = epoch % max(1, args.full_val_every) == 0 or epoch == args.epochs
        validation_kind = "skipped"
        if should_validate:
            validation_loader = val_loader if should_full_validate or quick_val_loader is None else quick_val_loader
            validation_kind = "full" if validation_loader is val_loader else "quick"
            val_metrics = validate_multitask(
                ema.module,
                validation_loader,
                args.data_root,
                device,
                binary_segmentation_bce_loss,
                validation_classification_criterion,
                seg_threshold=args.validation_threshold,
            )
        else:
            val_metrics = {
                "automated_score": 0.0,
                "segmentation_score": 0.0,
                "mean_iou": 0.0,
                "binary_foreground_iou": 0.0,
                "oracle_semantic_miou": 0.0,
                "boundary_f_score": 0.0,
                "rare_class_miou": 0.0,
                "classification_macro_accuracy": 0.0,
            }
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        row: dict[str, object] = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "validation_kind": validation_kind,
            "seg_threshold": args.validation_threshold,
            "learning_rate": lr,
        }
        history.append(row)
        print(
            "  "
            f"sup_cls={float(row['supervised_cls_loss']):.4f} "
            f"sup_seg={float(row['supervised_seg_loss']):.4f} "
            f"unsup={float(row['unsupervised_cls_loss']):.4f} "
            f"accept={float(row['pseudo_acceptance_rate']):.3f} "
            f"conf={float(row['pseudo_mean_confidence']):.3f} "
            f"coverage={int(float(row['pseudo_class_coverage']))}/{NUM_CLASSES} "
            f"val_auto={float(row['automated_score']):.4f} "
            f"mIoU={float(row['mean_iou']):.4f} "
            f"bin_iou={float(row['binary_foreground_iou']):.4f} "
            f"oracle_mIoU={float(row['oracle_semantic_miou']):.4f} "
            f"rare_mIoU={float(row['rare_class_miou']):.4f} "
            f"macro_acc={float(row['classification_macro_accuracy']):.4f} "
            f"lr={lr:.6f}"
        )
        print(f"  pseudo_top_hist={row['pseudo_top_hist']}")

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
        if validation_kind == "full" and float(row["automated_score"]) > best_score:
            best_score = float(row["automated_score"])
            save_checkpoint(
                args.checkpoint_dir / "best_multitask.pt",
                ema.module,
                optimizer,
                scheduler,
                epoch,
                best_score,
                args,
                source_args,
            )
            print(f"  saved new best checkpoint with selection_metric={best_score:.4f}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.checkpoint_dir / "multitask_fixmatch_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nWrote history: {history_path}")


if __name__ == "__main__":
    main()
