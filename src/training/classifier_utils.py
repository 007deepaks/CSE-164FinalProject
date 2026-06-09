"""Shared classifier checkpoint loading, ensembling, and TTA helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models.classification_model import build_classification_model, parse_depths


def classifier_model_kwargs_from_args(saved_args: dict[str, object]) -> dict[str, object]:
    return {
        "num_classes": 300,
        "base_channels": int(saved_args.get("base_channels", 48)),
        "depths": parse_depths(str(saved_args.get("depths", "2,2,4,2"))),
        "mlp_ratio": int(saved_args.get("mlp_ratio", 4)),
        "drop_path": float(saved_args.get("drop_path", 0.0)),
    }


def load_classifier_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_classification_model(**classifier_model_kwargs_from_args(checkpoint.get("args", {}))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_classifier_checkpoints(checkpoint_paths: list[Path] | None, device: torch.device) -> list[nn.Module]:
    if not checkpoint_paths:
        return []
    return [load_classifier_checkpoint(path, device) for path in checkpoint_paths]


def tta_views(images: torch.Tensor, tta: str) -> list[torch.Tensor]:
    if tta == "none":
        return [images]
    if tta == "hflip":
        return [images, torch.flip(images, dims=(-1,))]
    if tta == "multi_crop":
        image_size = images.shape[-1]
        up_size = int(round(image_size * 1.10))
        upsampled = F.interpolate(images, size=(up_size, up_size), mode="bilinear", align_corners=False)
        max_offset = up_size - image_size
        crop_boxes = [
            (0, 0),
            (0, max_offset),
            (max_offset, 0),
            (max_offset, max_offset),
            (max_offset // 2, max_offset // 2),
        ]
        crops = [
            upsampled[:, :, top : top + image_size, left : left + image_size]
            for top, left in crop_boxes
        ]
        crops.append(torch.flip(crops[-1], dims=(-1,)))
        return crops
    raise ValueError(f"Unsupported TTA mode: {tta}")


@torch.no_grad()
def classifier_logits_with_tta(
    models: list[nn.Module],
    images: torch.Tensor,
    tta: str = "none",
) -> torch.Tensor:
    if not models:
        raise ValueError("At least one classifier model is required")
    logits: list[torch.Tensor] = []
    for view in tta_views(images, tta):
        for model in models:
            logits.append(model(view))
    return torch.stack(logits, dim=0).mean(dim=0)
