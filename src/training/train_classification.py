"""Train a from-scratch image classifier for CSE164 class_id prediction."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.classification_dataset import BalancedClassBatchSampler, ClassificationDataset
from src.metrics.classification_metrics import ClassificationMetricTracker
from src.models.classification_model import build_classification_model, parse_depths


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum(-targets * F.log_softmax(logits, dim=1), dim=1))


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def smooth_one_hot(targets: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    with torch.no_grad():
        off_value = smoothing / num_classes
        on_value = 1.0 - smoothing + off_value
        y = torch.full((targets.shape[0], num_classes), off_value, device=targets.device)
        y.scatter_(1, targets.unsqueeze(1), on_value)
    return y


def rand_bbox(width: int, height: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = (1.0 - lam) ** 0.5
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    center_x = int(torch.randint(0, width, (1,)).item())
    center_y = int(torch.randint(0, height, (1,)).item())
    x1 = max(center_x - cut_width // 2, 0)
    y1 = max(center_y - cut_height // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y2 = min(center_y + cut_height // 2, height)
    return x1, y1, x2, y2


def maybe_apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_probs = smooth_one_hot(targets, num_classes, label_smoothing)
    if mix_prob <= 0 or torch.rand(1).item() > mix_prob or len(images) < 2:
        return images, target_probs
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or torch.rand(1).item() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha <= 0:
        return images, target_probs
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(images.shape[0], device=images.device)
    mixed_targets = lam * target_probs + (1.0 - lam) * target_probs[permutation]
    if use_cutmix:
        _, _, height, width = images.shape
        x1, y1, x2, y2 = rand_bbox(width, height, lam)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        box_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - box_area / float(width * height)
        mixed_targets = lam * target_probs + (1.0 - lam) * target_probs[permutation]
        return mixed_images, mixed_targets
    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    return mixed_images, mixed_targets


def random_erasing(images: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return images
    images = images.clone()
    _, _, height, width = images.shape
    for index in range(images.shape[0]):
        if torch.rand(1).item() > probability:
            continue
        erase_area = float(torch.empty(1).uniform_(0.02, 0.20).item()) * height * width
        aspect = float(torch.empty(1).uniform_(0.3, 3.3).item())
        erase_height = min(height, max(1, int(round((erase_area * aspect) ** 0.5))))
        erase_width = min(width, max(1, int(round((erase_area / aspect) ** 0.5))))
        top = int(torch.randint(0, max(1, height - erase_height + 1), (1,)).item())
        left = int(torch.randint(0, max(1, width - erase_width + 1), (1,)).item())
        images[index, :, top : top + erase_height, left : left + erase_width] = 0.0
    return images


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    args: argparse.Namespace,
    ema: ModelEma | None,
) -> float:
    model.train()
    running_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["class_id"].to(device, non_blocking=True)
        images = random_erasing(images, args.random_erasing_prob)
        images, training_targets = maybe_apply_mixup_cutmix(
            images,
            targets,
            num_classes=300,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, training_targets)
        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        running_loss += float(loss.item())
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  train batch {batch_index:04d}/{len(loader)} loss={loss.item():.4f}")
    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    tracker = ClassificationMetricTracker(num_classes=300)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["class_id"].to(device, non_blocking=True)
        logits = model(images)
        running_loss += float(criterion(logits, targets).item())
        tracker.update(logits, targets)
    metrics = tracker.compute()
    metrics["loss"] = running_loss / max(1, len(loader))
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_macro_accuracy: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_macro_accuracy": best_macro_accuracy,
            "args": safe_args,
        },
        path,
    )


def read_resume_args(checkpoint_path: Path | None, device: torch.device) -> dict[str, object]:
    if checkpoint_path is None:
        return {}
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint.get("args", {})


def load_resume_checkpoint(model: nn.Module, checkpoint_path: Path | None, device: torch.device) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded classifier checkpoint: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=0.0)
    parser.add_argument("--random-erasing-prob", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--augment-policy", choices=["basic", "strong"], default="strong")
    parser.add_argument("--split", choices=["train_labeled", "train_combined"], default="train_labeled")
    parser.add_argument("--include-seg-crops", action="store_true")
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--balanced-class-batches", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--depths", type=str, default="2,2,4,2")
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    resume_args = read_resume_args(args.resume_checkpoint, device)
    for key in ["base_channels", "depths", "mlp_ratio", "drop_path"]:
        if key in resume_args:
            setattr(args, key, resume_args[key])
    args.base_channels = int(args.base_channels)
    args.mlp_ratio = int(args.mlp_ratio)
    args.drop_path = float(args.drop_path)
    depths = parse_depths(args.depths)
    print(f"Using device: {device}; mixed precision: {use_amp}")
    print(
        "Classifier: ConvNeXt-style from scratch "
        f"base_channels={args.base_channels}, depths={depths}, mlp_ratio={args.mlp_ratio}"
    )

    train_dataset = ClassificationDataset(
        args.data_root,
        args.split,
        args.image_size,
        args.max_train_samples,
        augment=not args.no_augment,
        random_crop=not args.no_random_crop,
        augment_policy=args.augment_policy,
        include_seg_crops=args.include_seg_crops,
        crop_padding=args.crop_padding,
    )
    val_dataset = ClassificationDataset(args.data_root, "val", args.image_size, args.max_val_samples)
    if args.balanced_class_batches:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=BalancedClassBatchSampler(train_dataset.samples, args.batch_size),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Train samples: {len(train_dataset)}; val samples: {len(val_dataset)}")

    model = build_classification_model(
        num_classes=300,
        base_channels=args.base_channels,
        depths=depths,
        mlp_ratio=args.mlp_ratio,
        drop_path=args.drop_path,
    ).to(device)
    load_resume_checkpoint(model, args.resume_checkpoint, device)
    criterion = SoftTargetCrossEntropy()
    validation_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler_t_max = max(2, args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=args.min_learning_rate,
    )
    scaler = GradScaler(enabled=use_amp)
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None
    best_macro_accuracy = -1.0
    best_ema_macro_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, args, ema)
        val_metrics = validate(model, val_loader, validation_criterion, device)
        ema_metrics = validate(ema.module, val_loader, validation_criterion, device) if ema is not None else None
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
            "val_macro_accuracy": val_metrics["macro_accuracy"],
            "val_mean_confidence": val_metrics["mean_confidence"],
            "learning_rate": current_lr,
        }
        if ema_metrics is not None:
            row.update(
                {
                    "ema_val_loss": ema_metrics["loss"],
                    "ema_val_accuracy": ema_metrics["accuracy"],
                    "ema_val_top5_accuracy": ema_metrics["top5_accuracy"],
                    "ema_val_macro_accuracy": ema_metrics["macro_accuracy"],
                    "ema_val_mean_confidence": ema_metrics["mean_confidence"],
                }
            )
        history.append(row)
        print(
            "  "
            f"train_loss={train_loss:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"acc={row['val_accuracy']:.4f} "
            f"top5={row['val_top5_accuracy']:.4f} "
            f"macro_acc={row['val_macro_accuracy']:.4f} "
            f"conf={row['val_mean_confidence']:.4f} "
            f"lr={current_lr:.6f}"
        )
        if ema_metrics is not None:
            print(
                "  "
                f"ema_loss={row['ema_val_loss']:.4f} "
                f"ema_acc={row['ema_val_accuracy']:.4f} "
                f"ema_top5={row['ema_val_top5_accuracy']:.4f} "
                f"ema_macro={row['ema_val_macro_accuracy']:.4f} "
                f"ema_conf={row['ema_val_mean_confidence']:.4f}"
            )
        save_checkpoint(
            args.checkpoint_dir / "latest_classification.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_macro_accuracy,
            args,
        )
        if row["val_macro_accuracy"] > best_macro_accuracy:
            best_macro_accuracy = row["val_macro_accuracy"]
            save_checkpoint(
                args.checkpoint_dir / "best_classification.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_macro_accuracy,
                args,
            )
            print(f"  saved new best checkpoint with macro_acc={best_macro_accuracy:.4f}")
        if ema is not None and row["ema_val_macro_accuracy"] > best_ema_macro_accuracy:
            best_ema_macro_accuracy = row["ema_val_macro_accuracy"]
            save_checkpoint(
                args.checkpoint_dir / "best_ema_classification.pt",
                ema.module,
                optimizer,
                scheduler,
                epoch,
                best_ema_macro_accuracy,
                args,
            )
            print(f"  saved new best EMA checkpoint with macro_acc={best_ema_macro_accuracy:.4f}")

    history_path = args.checkpoint_dir / "classification_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nWrote history: {history_path}")


if __name__ == "__main__":
    main()
