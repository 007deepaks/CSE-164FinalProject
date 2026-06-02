"""Train a from-scratch image classifier for CSE164 class_id prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.classification_dataset import ClassificationDataset
from src.metrics.classification_metrics import ClassificationMetricTracker
from src.models.classification_model import build_classification_model, parse_depths


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["class_id"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
    depths = parse_depths(args.depths)
    print(f"Using device: {device}; mixed precision: {use_amp}")
    print(
        "Classifier: ConvNeXt-style from scratch "
        f"base_channels={args.base_channels}, depths={depths}, mlp_ratio={args.mlp_ratio}"
    )

    train_dataset = ClassificationDataset(args.data_root, "train_labeled", args.image_size, args.max_train_samples)
    val_dataset = ClassificationDataset(args.data_root, "val", args.image_size, args.max_val_samples)
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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler_t_max = max(2, args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=args.min_learning_rate,
    )
    scaler = GradScaler(enabled=use_amp)
    best_macro_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_accuracy": val_metrics["macro_accuracy"],
            "learning_rate": current_lr,
        }
        history.append(row)
        print(
            "  "
            f"train_loss={train_loss:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"acc={row['val_accuracy']:.4f} "
            f"macro_acc={row['val_macro_accuracy']:.4f} "
            f"lr={current_lr:.6f}"
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

    history_path = args.checkpoint_dir / "classification_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nWrote history: {history_path}")


if __name__ == "__main__":
    main()
