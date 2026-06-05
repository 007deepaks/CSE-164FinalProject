"""Train a shared ConvNeXt multi-task classifier and binary segmenter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.classification_dataset import ClassificationDataset
from src.data.segmentation_dataset import SegmentationDataset
from src.models.multitask_model import MODEL_CONFIGS, build_multitask_model, resolve_model_config
from src.training.multitask_utils import args_to_dict, validate_multitask
from src.utils.masks import IGNORE_ID


def dice_loss(segmentation_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_ID
    foreground = (target == 1).float()
    foreground_probability = torch.softmax(segmentation_logits, dim=1)[:, 1]
    valid_float = valid.float()
    intersection = (foreground_probability * foreground * valid_float).sum()
    denominator = ((foreground_probability + foreground) * valid_float).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


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
) -> dict[str, float]:
    model.train()
    totals = {
        "segmentation_ce_loss": 0.0,
        "segmentation_dice_loss": 0.0,
        "segmentation_classification_loss": 0.0,
        "segmentation_total_loss": 0.0,
    }
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            ce = segmentation_criterion(outputs["segmentation"], masks)
            dice = dice_loss(outputs["segmentation"], masks)
            class_loss = classification_criterion(outputs["classification"], class_ids)
            loss = (
                segmentation_loss_weight * ce
                + dice_loss_weight * dice
                + seg_classification_loss_weight * class_loss
            )
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        totals["segmentation_ce_loss"] += float(ce.item())
        totals["segmentation_dice_loss"] += float(dice.item())
        totals["segmentation_classification_loss"] += float(class_loss.item())
        totals["segmentation_total_loss"] += float(loss.item())
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  seg batch {batch_index:04d}/{len(loader)} loss={loss.item():.4f}")
    return {key: value / max(1, len(loader)) for key, value in totals.items()}


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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            raw_loss = classification_criterion(outputs["classification"], class_ids)
            loss = cls_loss_weight * raw_loss
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(raw_loss.item())
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  cls batch {batch_index:04d}/{len(loader)} loss={raw_loss.item():.4f}")
    return {"classification_train_loss": total_loss / max(1, len(loader))}


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
            "args": args_to_dict(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seg-batch-size", type=int, default=2)
    parser.add_argument("--cls-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--foreground-weight", type=float, default=1.0)
    parser.add_argument("--segmentation-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--seg-classification-loss-weight", type=float, default=0.5)
    parser.add_argument("--cls-loss-weight", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="small")
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--depths", type=str)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--drop-path", type=float, default=0.05)
    parser.add_argument("--decoder-channels", type=int)
    parser.add_argument("--max-seg-samples", type=int)
    parser.add_argument("--max-cls-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        augment=True,
    )
    cls_train = ClassificationDataset(
        args.data_root,
        split="train_labeled",
        image_size=args.image_size,
        max_samples=args.max_cls_samples,
        augment=True,
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

    model = build_multitask_model(
        model_size=args.model_size,
        base_channels=args.base_channels,
        depths=args.depths,
        mlp_ratio=args.mlp_ratio,
        drop_path=args.drop_path,
        decoder_channels=args.decoder_channels,
    ).to(device)
    seg_weights = torch.tensor(
        [args.background_weight, args.foreground_weight],
        dtype=torch.float32,
        device=device,
    )
    segmentation_criterion = nn.CrossEntropyLoss(weight=seg_weights, ignore_index=IGNORE_ID)
    classification_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(2, args.epochs),
        eta_min=args.min_learning_rate,
    )
    scaler = GradScaler(enabled=use_amp)
    best_score = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        seg_metrics = train_segmentation_batches(
            model,
            seg_loader,
            segmentation_criterion,
            classification_criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            args.segmentation_loss_weight,
            args.dice_loss_weight,
            args.seg_classification_loss_weight,
            args.gradient_clip,
        )
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
        )
        val_metrics = validate_multitask(
            model,
            val_loader,
            args.data_root,
            device,
            segmentation_criterion,
            classification_criterion,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": float(epoch),
            **seg_metrics,
            **cls_metrics,
            **val_metrics,
            "learning_rate": float(current_lr),
        }
        history.append(row)
        print(
            "  "
            f"seg_loss={row['segmentation_total_loss']:.4f} "
            f"cls_loss={row['classification_train_loss']:.4f} "
            f"val_auto={row['automated_score']:.4f} "
            f"val_seg={row['segmentation_score']:.4f} "
            f"mIoU={row['mean_iou']:.4f} "
            f"boundary={row['boundary_f_score']:.4f} "
            f"rare_mIoU={row['rare_class_miou']:.4f} "
            f"macro_acc={row['classification_macro_accuracy']:.4f} "
            f"lr={current_lr:.6f}"
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
        if row["automated_score"] > best_score:
            best_score = row["automated_score"]
            save_checkpoint(
                args.checkpoint_dir / "best_multitask.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                args,
            )
            print(f"  saved new best checkpoint with automated_score={best_score:.4f}")

    history_path = args.checkpoint_dir / "multitask_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nWrote history: {history_path}")


if __name__ == "__main__":
    main()
