"""Fine-tune the classifier with FixMatch on train_unlabeled images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.classification_dataset import (
    BalancedClassBatchSampler,
    ClassificationDataset,
    UnlabeledFixMatchDataset,
)
from src.metrics.classification_metrics import ClassificationMetricTracker
from src.models.classification_model import build_classification_model, parse_depths


def checkpoint_args_to_model_kwargs(saved_args: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    return {
        "num_classes": 300,
        "base_channels": int(saved_args.get("base_channels", args.base_channels)),
        "depths": parse_depths(str(saved_args.get("depths", args.depths))),
        "mlp_ratio": int(saved_args.get("mlp_ratio", args.mlp_ratio)),
        "drop_path": float(saved_args.get("drop_path", args.drop_path)),
    }


def load_or_build_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    checkpoint_args: dict[str, object] = {}
    model_kwargs: dict[str, object]
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        checkpoint_args = checkpoint.get("args", {})
        model_kwargs = checkpoint_args_to_model_kwargs(checkpoint_args, args)
        model = build_classification_model(**model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded classifier checkpoint: {args.resume_checkpoint}")
    else:
        model_kwargs = {
            "num_classes": 300,
            "base_channels": args.base_channels,
            "depths": parse_depths(args.depths),
            "mlp_ratio": args.mlp_ratio,
            "drop_path": args.drop_path,
        }
        model = build_classification_model(**model_kwargs).to(device)
    print(
        "Classifier: ConvNeXt-style from scratch "
        f"base_channels={model_kwargs['base_channels']}, "
        f"depths={model_kwargs['depths']}, "
        f"mlp_ratio={model_kwargs['mlp_ratio']}"
    )
    return model, checkpoint_args


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
    source_checkpoint_args: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    for key in ["base_channels", "depths", "mlp_ratio", "drop_path"]:
        if key in source_checkpoint_args:
            safe_args[key] = source_checkpoint_args[key]
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


def train_one_epoch(
    model: nn.Module,
    supervised_loader: DataLoader,
    unlabeled_loader: DataLoader,
    supervised_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    unlabeled_iter = infinite_loader(unlabeled_loader)
    supervised_iter = infinite_loader(supervised_loader)
    steps = args.steps_per_epoch or len(supervised_loader)
    running_supervised_loss = 0.0
    running_unsupervised_loss = 0.0
    accepted = 0
    unlabeled_seen = 0
    tracker = ClassificationMetricTracker(num_classes=300)

    for step in range(1, steps + 1):
        supervised_batch = next(supervised_iter)
        unlabeled_batch = next(unlabeled_iter)
        supervised_images = supervised_batch["image"].to(device, non_blocking=True)
        supervised_targets = supervised_batch["class_id"].to(device, non_blocking=True)
        weak_images = unlabeled_batch["weak_image"].to(device, non_blocking=True)
        strong_images = unlabeled_batch["strong_image"].to(device, non_blocking=True)

        with torch.no_grad():
            weak_logits_parts: list[torch.Tensor] = []
            weak_chunk_size = args.weak_forward_batch_size or len(weak_images)
            for weak_chunk in weak_images.split(weak_chunk_size):
                with autocast(device_type="cuda", enabled=use_amp):
                    weak_logits_parts.append(model(weak_chunk).float())
            weak_logits = torch.cat(weak_logits_parts, dim=0)
            pseudo_probabilities = torch.softmax(weak_logits.float(), dim=1)
            pseudo_confidence, pseudo_targets = pseudo_probabilities.max(dim=1)
            confidence_mask = pseudo_confidence.ge(args.confidence_threshold)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            supervised_logits = model(supervised_images)
            supervised_loss = supervised_criterion(supervised_logits, supervised_targets)
            if confidence_mask.any():
                accepted_strong_images = strong_images[confidence_mask]
                accepted_targets = pseudo_targets[confidence_mask]
                accepted_logits_parts: list[torch.Tensor] = []
                strong_chunk_size = args.strong_forward_batch_size or len(accepted_strong_images)
                for strong_chunk in accepted_strong_images.split(strong_chunk_size):
                    accepted_logits_parts.append(model(strong_chunk))
                strong_logits = torch.cat(accepted_logits_parts, dim=0)
                unsupervised_loss = nn.functional.cross_entropy(strong_logits, accepted_targets)
            else:
                unsupervised_loss = supervised_logits.sum() * 0.0
            loss = supervised_loss + args.unlabeled_loss_weight * unsupervised_loss

        if not torch.isfinite(loss):
            print(
                f"  skipped non-finite FixMatch step {step}: "
                f"sup={float(supervised_loss.detach().cpu()):.4f} "
                f"unsup={float(unsupervised_loss.detach().cpu()):.4f}"
            )
            continue

        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_supervised_loss += float(supervised_loss.detach().cpu().item())
        running_unsupervised_loss += float(unsupervised_loss.detach().cpu().item())
        accepted += int(confidence_mask.sum().item())
        unlabeled_seen += int(confidence_mask.numel())
        tracker.update(supervised_logits, supervised_targets)
        if step % args.print_every == 0 or step == steps:
            accept_rate = accepted / max(1, unlabeled_seen)
            print(
                f"  step {step:04d}/{steps} "
                f"sup={supervised_loss.item():.4f} "
                f"unsup={unsupervised_loss.item():.4f} "
                f"accept={accept_rate:.3f}"
            )

    train_metrics = tracker.compute()
    return {
        "supervised_loss": running_supervised_loss / max(1, steps),
        "unsupervised_loss": running_unsupervised_loss / max(1, steps),
        "pseudo_acceptance_rate": accepted / max(1, unlabeled_seen),
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_accuracy": train_metrics["macro_accuracy"],
    }


def infinite_loader(loader: DataLoader):
    """Yield batches forever without caching them like itertools.cycle does."""
    while True:
        for batch in loader:
            yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unlabeled-batch-size", type=int, default=64)
    parser.add_argument("--weak-forward-batch-size", type=int)
    parser.add_argument("--strong-forward-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--unlabeled-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--split", choices=["train_labeled", "train_combined"], default="train_combined")
    parser.add_argument("--include-seg-crops", action="store_true")
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--balanced-class-batches", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--unlabeled-random-crop", action="store_true")
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--depths", type=str, default="3,3,9,3")
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-unlabeled-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints/classifier_fixmatch"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}; mixed precision: {use_amp}")

    supervised_dataset = ClassificationDataset(
        args.data_root,
        args.split,
        args.image_size,
        args.max_train_samples,
        augment=not args.no_augment,
        random_crop=not args.no_random_crop,
        include_seg_crops=args.include_seg_crops,
        crop_padding=args.crop_padding,
    )
    unlabeled_dataset = UnlabeledFixMatchDataset(
        args.data_root,
        args.image_size,
        args.max_unlabeled_samples,
        random_crop=args.unlabeled_random_crop,
    )
    val_dataset = ClassificationDataset(args.data_root, "val", args.image_size, args.max_val_samples)
    if args.balanced_class_batches:
        supervised_loader = DataLoader(
            supervised_dataset,
            batch_sampler=BalancedClassBatchSampler(supervised_dataset.samples, args.batch_size),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        supervised_loader = DataLoader(
            supervised_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.unlabeled_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Supervised samples: {len(supervised_dataset)}; "
        f"unlabeled samples: {len(unlabeled_dataset)}; val samples: {len(val_dataset)}"
    )

    model, source_checkpoint_args = load_or_build_model(args, device)
    supervised_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    validation_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(2, args.epochs),
        eta_min=args.min_learning_rate,
    )
    scaler = GradScaler(enabled=use_amp)
    best_macro_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model,
            supervised_loader,
            unlabeled_loader,
            supervised_criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            args,
        )
        val_metrics = validate(model, val_loader, validation_criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_accuracy": val_metrics["macro_accuracy"],
            "learning_rate": current_lr,
        }
        history.append(row)
        print(
            "  "
            f"sup_loss={row['supervised_loss']:.4f} "
            f"unsup_loss={row['unsupervised_loss']:.4f} "
            f"accept={row['pseudo_acceptance_rate']:.3f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_macro={row['val_macro_accuracy']:.4f} "
            f"lr={current_lr:.6f}"
        )
        save_checkpoint(
            args.checkpoint_dir / "latest_fixmatch_classification.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_macro_accuracy,
            args,
            source_checkpoint_args,
        )
        if row["val_macro_accuracy"] > best_macro_accuracy:
            best_macro_accuracy = row["val_macro_accuracy"]
            save_checkpoint(
                args.checkpoint_dir / "best_fixmatch_classification.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_macro_accuracy,
                args,
                source_checkpoint_args,
            )
            print(f"  saved new best checkpoint with macro_acc={best_macro_accuracy:.4f}")

    history_path = args.checkpoint_dir / "fixmatch_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nWrote history: {history_path}")


if __name__ == "__main__":
    main()
