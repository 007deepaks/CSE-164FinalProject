"""Train a shared ConvNeXt multi-task classifier and binary segmenter."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.classification_dataset import BalancedClassBatchSampler, ClassificationDataset
from src.data.segmentation_dataset import SegmentationDataset
from src.models.multitask_model import MODEL_CONFIGS, build_multitask_model, resolve_model_config
from src.training.multitask_utils import (
    args_to_dict,
    binary_segmentation_bce_loss,
    binary_prediction_from_logits,
    binary_segmentation_dice_loss,
    validate_multitask,
)
from src.utils.masks import IGNORE_ID, NUM_CLASSES


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


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


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(0, min(warmup_epochs, max(0, epochs - 1)))
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        cosine_epochs = max(1, epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch_index - warmup_epochs) / cosine_epochs))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_segmentation_batches(
    model: nn.Module,
    loader: DataLoader,
    segmentation_criterion: nn.Module,
    classification_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    segmentation_loss_weight: float,
    dice_loss_weight: float,
    seg_classification_loss_weight: float,
    gradient_clip: float,
    ema: ModelEma | None = None,
) -> dict[str, float]:
    model.train()
    totals = {
        "segmentation_ce_loss": 0.0,
        "segmentation_dice_loss": 0.0,
        "segmentation_classification_loss": 0.0,
        "segmentation_total_loss": 0.0,
        "segmentation_skipped_batches": 0.0,
    }
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images, seg=True)
            ce = segmentation_criterion(outputs["segmentation"], masks)
            class_loss = classification_criterion(outputs["classification"], class_ids)
        dice = binary_segmentation_dice_loss(outputs["segmentation"].float(), masks)
        loss = (
            segmentation_loss_weight * ce
            + dice_loss_weight * dice
            + seg_classification_loss_weight * class_loss
        )
        components = {
            "ce": ce,
            "dice": dice,
            "class_loss": class_loss,
            "loss": loss,
        }
        bad_components = [name for name, value in components.items() if not torch.isfinite(value.detach()).all()]
        if bad_components:
            totals["segmentation_skipped_batches"] += 1.0
            print(f"  WARNING: skipping seg batch {batch_index:04d}; non-finite {bad_components}")
            continue
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not torch.isfinite(grad_norm):
                totals["segmentation_skipped_batches"] += 1.0
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(f"  WARNING: skipping seg batch {batch_index:04d}; non-finite grad_norm")
                continue
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        totals["segmentation_ce_loss"] += float(ce.item())
        totals["segmentation_dice_loss"] += float(dice.item())
        totals["segmentation_classification_loss"] += float(class_loss.item())
        totals["segmentation_total_loss"] += float(loss.item())
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  seg batch {batch_index:04d}/{len(loader)} loss={loss.item():.4f}")
    divisor = max(1, len(loader) - int(totals["segmentation_skipped_batches"]))
    return {
        key: (value / divisor if key != "segmentation_skipped_batches" else value)
        for key, value in totals.items()
    }


def train_classification_batches(
    model: nn.Module,
    loader: DataLoader,
    classification_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    cls_loss_weight: float,
    gradient_clip: float,
    metric_prefix: str = "classification_train",
    ema: ModelEma | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    skipped_batches = 0
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images, seg=False)
            raw_loss = classification_criterion(outputs["classification"], class_ids)
            loss = cls_loss_weight * raw_loss
        if not torch.isfinite(raw_loss.detach()).all() or not torch.isfinite(loss.detach()).all():
            skipped_batches += 1
            print(f"  WARNING: skipping cls batch {batch_index:04d}; non-finite loss")
            continue
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not torch.isfinite(grad_norm):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(f"  WARNING: skipping cls batch {batch_index:04d}; non-finite grad_norm")
                continue
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        total_loss += float(raw_loss.item())
        predictions = torch.argmax(outputs["classification"].detach(), dim=1)
        correct += int((predictions == class_ids).sum().item())
        total += int(class_ids.numel())
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  cls batch {batch_index:04d}/{len(loader)} loss={raw_loss.item():.4f}")
    divisor = max(1, len(loader) - skipped_batches)
    return {
        f"{metric_prefix}_loss": total_loss / divisor,
        f"{metric_prefix}_accuracy": correct / max(1, total),
        f"{metric_prefix}_skipped_batches": float(skipped_batches),
    }


def train_joint_mixed_batches(
    model: nn.Module,
    seg_loader: DataLoader,
    cls_loader: DataLoader,
    segmentation_criterion: nn.Module,
    classification_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    segmentation_loss_weight: float,
    dice_loss_weight: float,
    seg_classification_loss_weight: float,
    cls_loss_weight: float,
    gradient_clip: float,
    ema: ModelEma | None = None,
) -> dict[str, float]:
    model.train()
    totals = {
        "segmentation_ce_loss": 0.0,
        "segmentation_dice_loss": 0.0,
        "segmentation_classification_loss": 0.0,
        "segmentation_total_loss": 0.0,
        "classification_train_loss": 0.0,
        "mixed_total_loss": 0.0,
        "seg_classification_train_accuracy": 0.0,
        "classification_train_accuracy": 0.0,
        "mixed_skipped_batches": 0.0,
    }
    seg_iter = iter(seg_loader)
    cls_iter = iter(cls_loader)
    steps = max(len(seg_loader), len(cls_loader))
    seg_class_correct = 0
    seg_class_total = 0
    cls_correct = 0
    cls_total = 0
    completed_steps = 0

    for batch_index in range(1, steps + 1):
        try:
            seg_batch = next(seg_iter)
        except StopIteration:
            seg_iter = iter(seg_loader)
            seg_batch = next(seg_iter)
        try:
            cls_batch = next(cls_iter)
        except StopIteration:
            cls_iter = iter(cls_loader)
            cls_batch = next(cls_iter)

        seg_images = seg_batch["image"].to(device, non_blocking=True)
        masks = seg_batch["mask"].to(device, non_blocking=True)
        seg_class_ids = seg_batch["class_id"].to(device, non_blocking=True)
        cls_images = cls_batch["image"].to(device, non_blocking=True)
        cls_class_ids = cls_batch["class_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            seg_outputs = model(seg_images, seg=True)
            cls_outputs = model(cls_images, seg=False)
            ce = segmentation_criterion(seg_outputs["segmentation"], masks)
            seg_class_loss = classification_criterion(seg_outputs["classification"], seg_class_ids)
            cls_loss = classification_criterion(cls_outputs["classification"], cls_class_ids)
        dice = binary_segmentation_dice_loss(seg_outputs["segmentation"].float(), masks)
        seg_total = segmentation_loss_weight * ce + dice_loss_weight * dice + seg_classification_loss_weight * seg_class_loss
        loss = seg_total + cls_loss_weight * cls_loss

        components = {
            "ce": ce,
            "dice": dice,
            "seg_class_loss": seg_class_loss,
            "cls_loss": cls_loss,
            "loss": loss,
        }
        bad_components = [name for name, value in components.items() if not torch.isfinite(value.detach()).all()]
        if bad_components:
            totals["mixed_skipped_batches"] += 1.0
            print(f"  WARNING: skipping mixed batch {batch_index:04d}; non-finite {bad_components}")
            continue

        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not torch.isfinite(grad_norm):
                totals["mixed_skipped_batches"] += 1.0
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(f"  WARNING: skipping mixed batch {batch_index:04d}; non-finite grad_norm")
                continue
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        completed_steps += 1
        totals["segmentation_ce_loss"] += float(ce.item())
        totals["segmentation_dice_loss"] += float(dice.item())
        totals["segmentation_classification_loss"] += float(seg_class_loss.item())
        totals["segmentation_total_loss"] += float(seg_total.item())
        totals["classification_train_loss"] += float(cls_loss.item())
        totals["mixed_total_loss"] += float(loss.item())

        seg_predictions = torch.argmax(seg_outputs["classification"].detach(), dim=1)
        cls_predictions = torch.argmax(cls_outputs["classification"].detach(), dim=1)
        seg_class_correct += int((seg_predictions == seg_class_ids).sum().item())
        seg_class_total += int(seg_class_ids.numel())
        cls_correct += int((cls_predictions == cls_class_ids).sum().item())
        cls_total += int(cls_class_ids.numel())
        if batch_index % 20 == 0 or batch_index == steps:
            print(
                f"  mixed batch {batch_index:04d}/{steps} "
                f"seg_loss={seg_total.item():.4f} cls_loss={cls_loss.item():.4f}"
            )

    divisor = max(1, completed_steps)
    averages = {
        key: (value / divisor if key not in {"mixed_skipped_batches"} else value)
        for key, value in totals.items()
    }
    averages["seg_classification_train_accuracy"] = seg_class_correct / max(1, seg_class_total)
    averages["classification_train_accuracy"] = cls_correct / max(1, cls_total)
    return averages


@torch.no_grad()
def compute_debug_train_metrics(
    model: nn.Module,
    seg_loader: DataLoader,
    cls_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Measure whether the model can memorize the tiny debug subsets."""
    model.eval()
    seg_intersection = 0
    seg_union = 0
    seg_valid_pixels = 0
    seg_correct_pixels = 0
    semantic_intersections = torch.zeros(NUM_CLASSES + 1, dtype=torch.float64)
    semantic_unions = torch.zeros(NUM_CLASSES + 1, dtype=torch.float64)
    seg_class_correct = 0
    seg_class_total = 0
    cls_correct = 0
    cls_total = 0
    predicted_foreground_pixels = 0
    target_foreground_pixels = 0

    for batch in seg_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        semantic_masks = batch["semantic_mask"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        outputs = model(images, seg=True)
        binary_prediction = binary_prediction_from_logits(outputs["segmentation"], None)
        class_prediction = torch.argmax(outputs["classification"], dim=1)
        valid = masks != IGNORE_ID
        pred_fg = (binary_prediction == 1) & valid
        target_fg = (masks == 1) & valid
        seg_intersection += int((pred_fg & target_fg).sum().item())
        seg_union += int((pred_fg | target_fg).sum().item())
        seg_correct_pixels += int(((binary_prediction == masks) & valid).sum().item())
        seg_valid_pixels += int(valid.sum().item())
        predicted_foreground_pixels += int(pred_fg.sum().item())
        target_foreground_pixels += int(target_fg.sum().item())
        seg_class_correct += int((class_prediction == class_ids).sum().item())
        seg_class_total += int(class_ids.numel())

        oracle_semantic = torch.zeros_like(binary_prediction)
        semantic_ids = (class_ids.long() + 1).view(-1, 1, 1)
        oracle_semantic = torch.where(binary_prediction == 1, semantic_ids.expand_as(binary_prediction), oracle_semantic)
        valid_semantic = semantic_masks != IGNORE_ID
        for segmentation_id in torch.unique(semantic_masks[valid_semantic]).detach().cpu().tolist():
            if int(segmentation_id) <= 0:
                continue
            pred_class = (oracle_semantic == int(segmentation_id)) & valid_semantic
            target_class = (semantic_masks == int(segmentation_id)) & valid_semantic
            semantic_intersections[int(segmentation_id)] += float((pred_class & target_class).sum().item())
            semantic_unions[int(segmentation_id)] += float((pred_class | target_class).sum().item())

    for batch in cls_loader:
        images = batch["image"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        outputs = model(images, seg=False)
        class_prediction = torch.argmax(outputs["classification"], dim=1)
        cls_correct += int((class_prediction == class_ids).sum().item())
        cls_total += int(class_ids.numel())

    present = semantic_unions > 0
    semantic_miou = float((semantic_intersections[present] / semantic_unions[present]).mean().item()) if present.any() else 0.0
    return {
        "debug_train_binary_fg_iou": seg_intersection / max(1, seg_union),
        "debug_train_binary_pixel_accuracy": seg_correct_pixels / max(1, seg_valid_pixels),
        "debug_train_oracle_semantic_miou": semantic_miou,
        "debug_train_seg_class_accuracy": seg_class_correct / max(1, seg_class_total),
        "debug_train_cls_accuracy": cls_correct / max(1, cls_total),
        "debug_train_predicted_foreground_pct": 100.0 * predicted_foreground_pixels / max(1, seg_valid_pixels),
        "debug_train_target_foreground_pct": 100.0 * target_foreground_pixels / max(1, seg_valid_pixels),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_automated_score": best_score,
            "args": {
                **args_to_dict(args),
                "mask_guided_classifier": not getattr(args, "no_mask_guided_classifier", False),
            },
        },
        path,
    )


def load_model_weights(path: Path, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model weights from {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--joint-epochs", type=int, default=24)
    parser.add_argument("--seg-batch-size", type=int, default=2)
    parser.add_argument("--cls-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--stage", choices=["classifier_warmup", "joint", "warmup_joint"], default="joint")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--foreground-weight", type=float, default=1.0)
    parser.add_argument("--segmentation-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--seg-classification-loss-weight", type=float, default=0.5)
    parser.add_argument("--cls-loss-weight", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="small")
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--depths", type=str)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--drop-path", type=float, default=0.05)
    parser.add_argument("--decoder-channels", type=int)
    parser.add_argument("--num-segmentation-classes", type=int, default=1)
    parser.add_argument("--decoder-type", choices=["unet", "fpn"], default="unet")
    parser.add_argument("--no-mask-guided-classifier", action="store_true")
    parser.add_argument("--max-seg-samples", type=int)
    parser.add_argument("--max-cls-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--balanced-class-batches", action="store_true")
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--full-val-every", type=int, default=1)
    parser.add_argument("--quick-val-samples", type=int)
    parser.add_argument("--validation-threshold", type=float, default=0.50)
    parser.add_argument(
        "--validation-thresholds",
        type=str,
        default=None,
        help="Deprecated during training; use post-training threshold sweep instead.",
    )
    parser.add_argument(
        "--no-random-crop",
        action="store_true",
        help="Disable random resized crop while keeping flip/color jitter augmentations.",
    )
    parser.add_argument(
        "--debug-overfit",
        action="store_true",
        help="Overfit 8 segmentation and 32 classification samples with augmentation disabled.",
    )
    return parser.parse_args()


def run_one_stage(args: argparse.Namespace) -> Path:
    if args.debug_overfit:
        args.max_seg_samples = 8
        args.max_cls_samples = 32
        args.max_val_samples = 32 if args.max_val_samples is None else args.max_val_samples
        args.num_workers = 0
        args.label_smoothing = 0.0
        args.weight_decay = 0.0
        args.drop_path = 0.0
        args.no_random_crop = True
        args.min_learning_rate = min(args.min_learning_rate, args.learning_rate)
        print("DEBUG OVERFIT: using 8 segmentation samples, 32 classification samples, no augmentation.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    resolved = resolve_model_config(args.model_size, args.base_channels, args.depths, args.decoder_channels)
    print(f"Using device: {device}; mixed precision: {use_amp}")
    print(
        "Multi-task ConvNeXt from scratch: "
        f"model_size={args.model_size}, image_size={args.image_size}, "
        f"base_channels={resolved['base_channels']}, depths={resolved['depths']}, "
        f"decoder_channels={resolved['decoder_channels']}"
    )

    seg_train = SegmentationDataset(
        args.data_root,
        split="train_seg",
        image_size=args.image_size,
        target_mode="binary",
        max_samples=args.max_seg_samples,
        augment=not args.debug_overfit,
        random_crop=not args.no_random_crop,
    )
    cls_train = ClassificationDataset(
        args.data_root,
        split="train_combined",
        image_size=args.image_size,
        max_samples=args.max_cls_samples,
        augment=not args.debug_overfit,
        random_crop=not args.no_random_crop,
    )
    val_dataset = SegmentationDataset(
        args.data_root,
        split="val",
        image_size=args.image_size,
        target_mode="binary",
        max_samples=args.max_val_samples,
        augment=False,
    )
    seg_loader = DataLoader(
        seg_train,
        batch_size=args.seg_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.balanced_class_batches:
        cls_batch_sampler = BalancedClassBatchSampler(cls_train.samples, args.cls_batch_size)
        cls_loader = DataLoader(
            cls_train,
            batch_sampler=cls_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        cls_loader = DataLoader(
            cls_train,
            batch_size=args.cls_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Train samples: segmentation={len(seg_train)}, classification={len(cls_train)}; "
        f"val={len(val_dataset)}"
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
        print(f"Quick validation samples: {len(quick_val_dataset)}")

    model = build_multitask_model(
        model_size=args.model_size,
        base_channels=args.base_channels,
        depths=args.depths,
        mlp_ratio=args.mlp_ratio,
        drop_path=args.drop_path,
        decoder_channels=args.decoder_channels,
        num_segmentation_classes=args.num_segmentation_classes,
        decoder_type=args.decoder_type,
        mask_guided_classifier=not args.no_mask_guided_classifier,
    ).to(device)
    if args.resume_checkpoint is not None:
        load_model_weights(args.resume_checkpoint, model, device)
    if args.num_segmentation_classes == 1:
        segmentation_criterion = binary_segmentation_bce_loss
    else:
        seg_weights = torch.tensor(
            [args.background_weight, args.foreground_weight],
            dtype=torch.float32,
            device=device,
        )
        segmentation_criterion = nn.CrossEntropyLoss(weight=seg_weights, ignore_index=IGNORE_ID)
    classification_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
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
        print(
            f"LR scheduler: linear warmup for {args.warmup_epochs} epochs "
            f"to {args.learning_rate:g}, then cosine decay to {args.min_learning_rate:g}"
        )
    else:
        print(f"LR scheduler: cosine decay from {args.learning_rate:g} to {args.min_learning_rate:g}")
    scaler = GradScaler(enabled=use_amp)
    ema = None if args.no_ema else ModelEma(model, args.ema_decay)
    best_score = -1.0
    history: list[dict[str, float]] = []
    print(f"Training stage: {args.stage}")
    if args.validation_thresholds is not None:
        print("WARNING: --validation-thresholds is ignored during training; using --validation-threshold.")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        if args.stage == "classifier_warmup":
            seg_metrics = {
                "segmentation_ce_loss": 0.0,
                "segmentation_dice_loss": 0.0,
                "segmentation_classification_loss": 0.0,
                "segmentation_total_loss": 0.0,
                "segmentation_skipped_batches": 0.0,
            }
            seg_cls_metrics = train_classification_batches(
                model,
                seg_loader,
                classification_criterion,
                optimizer,
                device,
                scaler,
                use_amp,
                args.seg_classification_loss_weight,
                args.gradient_clip,
                ema=ema,
                metric_prefix="seg_classification_train",
            )
        else:
            mixed_metrics = train_joint_mixed_batches(
                model,
                seg_loader,
                cls_loader,
                segmentation_criterion,
                classification_criterion,
                optimizer,
                device,
                scaler,
                use_amp,
                args.segmentation_loss_weight,
                args.dice_loss_weight,
                args.seg_classification_loss_weight,
                args.cls_loss_weight,
                args.gradient_clip,
                ema=ema,
            )
            seg_metrics = {
                "segmentation_ce_loss": mixed_metrics["segmentation_ce_loss"],
                "segmentation_dice_loss": mixed_metrics["segmentation_dice_loss"],
                "segmentation_classification_loss": mixed_metrics["segmentation_classification_loss"],
                "segmentation_total_loss": mixed_metrics["segmentation_total_loss"],
                "segmentation_skipped_batches": mixed_metrics["mixed_skipped_batches"],
            }
            seg_cls_metrics = {
                "seg_classification_train_loss": mixed_metrics["segmentation_classification_loss"],
                "seg_classification_train_accuracy": mixed_metrics["seg_classification_train_accuracy"],
                "seg_classification_train_skipped_batches": mixed_metrics["mixed_skipped_batches"],
            }
            cls_metrics = {
                "classification_train_loss": mixed_metrics["classification_train_loss"],
                "classification_train_accuracy": mixed_metrics["classification_train_accuracy"],
                "classification_train_skipped_batches": mixed_metrics["mixed_skipped_batches"],
            }
        if args.stage == "classifier_warmup":
            cls_metrics = train_classification_batches(
                model,
                cls_loader,
                classification_criterion,
                optimizer,
                device,
                scaler,
                use_amp,
                args.cls_loss_weight,
                args.gradient_clip,
                ema=ema,
                metric_prefix="classification_train",
            )
        should_validate = epoch % max(1, args.validate_every) == 0 or epoch == args.epochs
        should_full_validate = epoch % max(1, args.full_val_every) == 0 or epoch == args.epochs
        val_metrics = {}
        validation_kind = "skipped"
        if should_validate:
            validation_loader = val_loader if should_full_validate or quick_val_loader is None else quick_val_loader
            validation_kind = "full" if validation_loader is val_loader else "quick"
            validation_model = ema.module if ema is not None else model
            val_metrics = validate_multitask(
                validation_model,
                validation_loader,
                args.data_root,
                device,
                segmentation_criterion,
                classification_criterion,
                seg_threshold=args.validation_threshold,
            )
            val_metrics["seg_threshold"] = float(args.validation_threshold)
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
                "seg_threshold": 0.0,
            }
        debug_metrics = (
            compute_debug_train_metrics(model, seg_loader, cls_loader, device)
            if args.debug_overfit
            else {}
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": float(epoch),
            **seg_metrics,
            **seg_cls_metrics,
            **cls_metrics,
            **val_metrics,
            **debug_metrics,
            "learning_rate": float(current_lr),
            "validation_kind": validation_kind,
        }
        history.append(row)
        print(
            "  "
            f"seg_loss={row['segmentation_total_loss']:.4f} "
            f"cls_loss={row['classification_train_loss']:.4f} "
            f"train_cls_acc={row['classification_train_accuracy']:.4f} "
            f"train_seg_cls_acc={row['seg_classification_train_accuracy']:.4f} "
            f"val_auto={row['automated_score']:.4f} "
            f"val_seg={row['segmentation_score']:.4f} "
            f"mIoU={row['mean_iou']:.4f} "
            f"bin_iou={row['binary_foreground_iou']:.4f} "
            f"oracle_mIoU={row['oracle_semantic_miou']:.4f} "
            f"boundary={row['boundary_f_score']:.4f} "
            f"rare_mIoU={row['rare_class_miou']:.4f} "
            f"macro_acc={row['classification_macro_accuracy']:.4f} "
            f"thr={row['seg_threshold']:.2f} "
            f"lr={current_lr:.6f}"
        )
        if args.debug_overfit:
            print(
                "  DEBUG train "
                f"binary_fg_iou={row['debug_train_binary_fg_iou']:.4f} "
                f"binary_pix_acc={row['debug_train_binary_pixel_accuracy']:.4f} "
                f"oracle_sem_mIoU={row['debug_train_oracle_semantic_miou']:.4f} "
                f"seg_cls_acc={row['debug_train_seg_class_accuracy']:.4f} "
                f"cls_acc={row['debug_train_cls_accuracy']:.4f} "
                f"pred_fg={row['debug_train_predicted_foreground_pct']:.2f}% "
                f"target_fg={row['debug_train_target_foreground_pct']:.2f}%"
            )
        save_checkpoint(
            args.checkpoint_dir / "latest_multitask.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            args,
        )
        selection_metric = (
            row["classification_macro_accuracy"]
            if args.stage == "classifier_warmup"
            else row["automated_score"]
        )
        can_save_best = validation_kind == "full"
        if can_save_best and selection_metric > best_score:
            best_score = selection_metric
            save_checkpoint(
                args.checkpoint_dir / "best_multitask.pt",
                ema.module if ema is not None else model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                args,
            )
            print(f"  saved new best checkpoint with selection_metric={best_score:.4f}")

    history_path = args.checkpoint_dir / "multitask_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nWrote history: {history_path}")
    return args.checkpoint_dir / "best_multitask.pt"


def main() -> None:
    args = parse_args()
    if args.stage != "warmup_joint":
        run_one_stage(args)
        return

    base_dir = args.checkpoint_dir
    warmup_args = argparse.Namespace(**vars(args))
    warmup_args.stage = "classifier_warmup"
    warmup_args.epochs = args.warmup_epochs
    warmup_args.learning_rate = args.warmup_learning_rate
    warmup_args.checkpoint_dir = base_dir / "warmup"
    warmup_args.resume_checkpoint = None
    print("\n=== Warmup phase ===")
    warmup_checkpoint = run_one_stage(warmup_args)

    joint_args = argparse.Namespace(**vars(args))
    joint_args.stage = "joint"
    joint_args.epochs = args.joint_epochs
    joint_args.checkpoint_dir = base_dir / "joint"
    joint_args.resume_checkpoint = warmup_checkpoint
    print("\n=== Joint phase ===")
    run_one_stage(joint_args)


if __name__ == "__main__":
    main()
