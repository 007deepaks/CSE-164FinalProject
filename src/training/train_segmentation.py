"""Train the first supervised segmentation baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import IMAGE_MEAN, IMAGE_STD, SegmentationDataset
from src.metrics.segmentation_metrics import SegmentationMetricTracker
from src.models.segmentation_model import build_segmentation_model
from src.utils.masks import IGNORE_ID


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu() * IMAGE_STD + IMAGE_MEAN
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colors[..., 0] = ((mask * 37) % 255).astype(np.uint8)
    colors[..., 1] = ((mask * 91) % 255).astype(np.uint8)
    colors[..., 2] = ((mask * 151) % 255).astype(np.uint8)
    colors[mask == 0] = np.array([20, 20, 20], dtype=np.uint8)
    colors[mask == IGNORE_ID] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(colors, mode="RGB")


def save_prediction_panels(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    output_dir: Path,
    prefix: str,
    max_examples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = torch.argmax(logits.detach(), dim=1).cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    count = min(max_examples, images.shape[0])
    for index in range(count):
        image = Image.fromarray(denormalize_image(images[index]), mode="RGB")
        gt = colorize_mask(masks_np[index])
        pred = colorize_mask(predictions[index])
        width, height = image.size
        panel = Image.new("RGB", (width * 3, height), color=(0, 0, 0))
        panel.paste(image, (0, 0))
        panel.paste(gt, (width, 0))
        panel.paste(pred, (width * 2, 0))
        panel.save(output_dir / f"{prefix}_{index:03d}.jpg", quality=95)


def summarize_prediction_distribution(counter: Counter[int], top_k: int = 10) -> list[tuple[int, int, float]]:
    """Return the most common predicted ids as (id, pixels, percent)."""
    total = sum(counter.values())
    if total == 0:
        return []
    return [
        (mask_id, count, 100.0 * count / total)
        for mask_id, count in counter.most_common(top_k)
    ]


def update_valid_mask_counter(counter: Counter[int], mask: torch.Tensor) -> None:
    """Count mask ids, excluding ignore pixels."""
    valid_mask = mask.detach().cpu()
    valid_mask = valid_mask[valid_mask != IGNORE_ID]
    ids, counts = torch.unique(valid_mask, return_counts=True)
    counter.update({int(mask_id): int(count) for mask_id, count in zip(ids, counts)})


def foreground_percentage(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    foreground = sum(count for mask_id, count in counter.items() if mask_id > 0)
    return 100.0 * foreground / total


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
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
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
    figure_dir: Path,
    epoch: int,
    num_visualizations: int,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    tracker = SegmentationMetricTracker(ignore_index=IGNORE_ID)
    prediction_counter: Counter[int] = Counter()
    ground_truth_counter: Counter[int] = Counter()
    saved_visuals = False
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += float(loss.item())
        tracker.update(logits, masks)
        predictions = torch.argmax(logits.detach(), dim=1)
        ids, counts = torch.unique(predictions.cpu(), return_counts=True)
        prediction_counter.update({int(mask_id): int(count) for mask_id, count in zip(ids, counts)})
        update_valid_mask_counter(ground_truth_counter, masks)
        if not saved_visuals and num_visualizations > 0:
            save_prediction_panels(
                images.cpu(),
                masks.cpu(),
                logits.cpu(),
                figure_dir,
                prefix=f"epoch_{epoch:03d}_val",
                max_examples=num_visualizations,
            )
            saved_visuals = True
    metrics = tracker.compute()
    metrics["loss"] = running_loss / max(1, len(loader))
    print("  val foreground pixels:")
    print(f"    predicted foreground: {foreground_percentage(prediction_counter):6.2f}%")
    print(f"    ground truth foreground: {foreground_percentage(ground_truth_counter):6.2f}%")
    print("  val top predicted ids:")
    for mask_id, count, percent in summarize_prediction_distribution(prediction_counter):
        print(f"    id={mask_id:3d} pixels={count:10d} ({percent:6.2f}%)")
    print("  val top ground-truth ids:")
    for mask_id, count, percent in summarize_prediction_distribution(ground_truth_counter):
        print(f"    id={mask_id:3d} pixels={count:10d} ({percent:6.2f}%)")
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_miou: float,
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
            "best_miou": best_miou,
            "args": safe_args,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--foreground-weight", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--num-visualizations", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}; mixed precision: {use_amp}")

    train_dataset = SegmentationDataset(args.data_root, "train_seg", args.image_size, args.max_train_samples)
    val_dataset = SegmentationDataset(args.data_root, "val", args.image_size, args.max_val_samples)
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

    model = build_segmentation_model(num_classes=301, base_channels=args.base_channels).to(device)
    class_weights = torch.full((301,), float(args.foreground_weight), dtype=torch.float32, device=device)
    class_weights[0] = float(args.background_weight)
    print(
        "CrossEntropyLoss weights: "
        f"background={class_weights[0].item():g}, "
        f"foreground={class_weights[1].item():g}"
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler_t_max = max(2, args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=args.min_learning_rate,
    )
    print(
        "Scheduler: CosineAnnealingLR "
        f"T_max={scheduler_t_max}, eta_min={args.min_learning_rate:g}"
    )
    scaler = GradScaler(enabled=use_amp)
    best_miou = -1.0

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_metrics = validate(model, val_loader, criterion, device, args.figure_dir, epoch, args.num_visualizations)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "learning_rate": current_lr,
        }
        history.append(row)
        print(
            "  "
            f"train_loss={train_loss:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"pixel_acc={row['val_pixel_accuracy']:.4f} "
            f"mIoU={row['val_mean_iou']:.4f} "
            f"lr={current_lr:.6f}"
        )
        save_checkpoint(args.checkpoint_dir / "latest_segmentation.pt", model, optimizer, scheduler, epoch, best_miou, args)
        if row["val_mean_iou"] > best_miou:
            best_miou = row["val_mean_iou"]
            save_checkpoint(args.checkpoint_dir / "best_segmentation.pt", model, optimizer, scheduler, epoch, best_miou, args)
            print(f"  saved new best checkpoint with mIoU={best_miou:.4f}")

    history_path = args.checkpoint_dir / "segmentation_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nWrote history: {history_path}")


if __name__ == "__main__":
    main()
