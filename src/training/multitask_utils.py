"""Shared utilities for the ConvNeXt multi-task training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from starter.kaggle_metric import detailed_score, encode_mask_ids
from src.metrics.classification_metrics import ClassificationMetricTracker
from src.models.multitask_model import build_multitask_model
from src.models.multitaskResnet50 import build_resnet50_multitask_model
from src.training.classifier_utils import classifier_logits_with_tta
from src.utils.masks import IGNORE_ID, NUM_CLASSES, decode_rgb_mask, encode_mask_to_rle, validate_prediction_mask


def args_to_dict(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def checkpoint_model_kwargs(saved_args: dict[str, object]) -> dict[str, object]:
    if "mask_guided_classifier" in saved_args:
        mask_guided_classifier = _optional_bool(saved_args.get("mask_guided_classifier"), default=True)
    elif "no_mask_guided_classifier" in saved_args:
        mask_guided_classifier = not _optional_bool(saved_args.get("no_mask_guided_classifier"), default=False)
    else:
        mask_guided_classifier = False
    return {
        "model_size": str(saved_args.get("model_size", "small")),
        "base_channels": _optional_int(saved_args.get("base_channels")),
        "depths": _optional_str(saved_args.get("depths")),
        "mlp_ratio": int(saved_args.get("mlp_ratio", 4)),
        "drop_path": float(saved_args.get("drop_path", 0.0)),
        "decoder_channels": _optional_int(saved_args.get("decoder_channels")),
        "num_segmentation_classes": int(saved_args.get("num_segmentation_classes", 2)),
        "decoder_type": str(saved_args.get("decoder_type", "fpn")),
        "mask_guided_classifier": mask_guided_classifier,
    }


def load_multitask_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    architecture = str(saved_args.get("architecture", "convnext"))
    if architecture == "resnet50":
        model = build_resnet50_multitask_model(
            num_segmentation_classes=int(saved_args.get("num_segmentation_classes", 1)),
            dropout=float(saved_args.get("resnet_classifier_dropout", 0.2)),
        ).to(device)
    else:
        model = build_multitask_model(**checkpoint_model_kwargs(saved_args)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint, saved_args


def build_val_solution_frame(data_root: Path, image_names: set[str] | None = None) -> pd.DataFrame:
    class_rows = pd.read_json(data_root / "val" / "classification.json")
    class_by_image = {str(row.image): int(row.class_id) for row in class_rows.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for image_path in sorted((data_root / "val" / "images").glob("*.JPEG")):
        if image_names is not None and image_path.name not in image_names:
            continue
        mask_path = data_root / "val" / "masks" / image_path.with_suffix(".png").name
        with Image.open(image_path) as image:
            width, height = image.size
        rows.append(
            {
                "image": image_path.name,
                "height": height,
                "width": width,
                "class_id": class_by_image[image_path.name],
                "segmentation_rle": encode_mask_ids(decode_rgb_mask(mask_path)),
            }
        )
    return pd.DataFrame(rows)


def semantic_mask_from_logits(
    segmentation_logits: torch.Tensor,
    classification_logits: torch.Tensor,
    index: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, int]:
    resized_logits = F.interpolate(
        segmentation_logits[index : index + 1],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    binary_prediction = binary_prediction_from_logits(resized_logits, None).squeeze(0).cpu().numpy()
    class_id = int(torch.argmax(classification_logits[index]).detach().cpu().item())
    mask = np.where(binary_prediction == 1, class_id + 1, 0).astype(np.uint16)
    validate_prediction_mask(mask)
    return mask, class_id


def foreground_probability_from_logits(segmentation_logits: torch.Tensor) -> torch.Tensor:
    if segmentation_logits.shape[1] == 1:
        return torch.sigmoid(segmentation_logits[:, 0])
    return torch.softmax(segmentation_logits, dim=1)[:, 1]


def binary_prediction_from_logits(
    segmentation_logits: torch.Tensor,
    seg_threshold: float | None = None,
) -> torch.Tensor:
    if segmentation_logits.shape[1] == 1:
        threshold = 0.5 if seg_threshold is None else seg_threshold
        return (torch.sigmoid(segmentation_logits[:, 0]) > threshold).long()
    if seg_threshold is None:
        return torch.argmax(segmentation_logits, dim=1)
    return (torch.softmax(segmentation_logits, dim=1)[:, 1] > seg_threshold).long()


def binary_segmentation_bce_loss(segmentation_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if segmentation_logits.shape[1] != 1:
        return F.cross_entropy(segmentation_logits, target, ignore_index=IGNORE_ID)
    valid = target != IGNORE_ID
    if not valid.any():
        return segmentation_logits.sum() * 0.0
    binary_target = ((target > 0) & valid).float()
    loss = F.binary_cross_entropy_with_logits(
        segmentation_logits[:, 0],
        binary_target,
        reduction="none",
    )
    return loss[valid].mean()


def binary_segmentation_dice_loss(segmentation_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_ID
    foreground = ((target > 0) & valid).float()
    foreground_probability = foreground_probability_from_logits(segmentation_logits)
    valid_float = valid.float()
    intersection = (foreground_probability * foreground * valid_float).sum()
    denominator = ((foreground_probability + foreground) * valid_float).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def semantic_mask_from_binary_and_class_logits(
    segmentation_logits: torch.Tensor,
    classification_logits: torch.Tensor,
    index: int,
    height: int,
    width: int,
    seg_threshold: float | None = None,
) -> tuple[np.ndarray, int]:
    resized_logits = F.interpolate(
        segmentation_logits[index : index + 1],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    binary_prediction = binary_prediction_from_logits(resized_logits, seg_threshold).squeeze(0).cpu().numpy()
    if binary_prediction.sum() == 0:
        foreground_probability = foreground_probability_from_logits(resized_logits).squeeze(0)
        flat_index = int(torch.argmax(foreground_probability).cpu().item())
        y, x = divmod(flat_index, width)
        binary_prediction[y, x] = 1
    class_id = int(torch.argmax(classification_logits[index]).detach().cpu().item())
    mask = np.where(binary_prediction == 1, class_id + 1, 0).astype(np.uint16)
    validate_prediction_mask(mask)
    return mask, class_id


@torch.no_grad()
def validate_multitask(
    model: nn.Module,
    loader: DataLoader,
    data_root: Path,
    device: torch.device,
    segmentation_criterion: nn.Module | None = None,
    classification_criterion: nn.Module | None = None,
    classifier_models: list[nn.Module] | None = None,
    tta: str = "none",
    seg_threshold: float | None = None,
) -> dict[str, float]:
    model.eval()
    classifier_models = classifier_models or []
    for classifier_model in classifier_models:
        classifier_model.eval()
    running_segmentation_loss = 0.0
    running_classification_loss = 0.0
    loss_batches = 0
    class_tracker = ClassificationMetricTracker(num_classes=NUM_CLASSES)
    submission_rows: list[dict[str, object]] = []
    binary_intersection = 0
    binary_union = 0
    binary_correct_pixels = 0
    binary_valid_pixels = 0
    oracle_intersections = np.zeros(NUM_CLASSES + 1, dtype=np.float64)
    oracle_unions = np.zeros(NUM_CLASSES + 1, dtype=np.float64)
    predicted_foreground_pixels = 0
    target_foreground_pixels = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        semantic_masks = batch["semantic_mask"].to(device, non_blocking=True)
        class_ids = batch["class_id"].to(device, non_blocking=True)
        outputs = model(images)
        segmentation_logits = outputs["segmentation"]
        classification_logits = outputs["classification"]
        if classifier_models:
            classification_logits = classifier_logits_with_tta(classifier_models, images, tta)
        if tta in {"hflip", "multi_crop"}:
            flipped_images = torch.flip(images, dims=(-1,))
            flipped_outputs = model(flipped_images)
            flipped_classification_logits = flipped_outputs["classification"]
            if not classifier_models:
                classification_logits = 0.5 * (classification_logits + flipped_classification_logits)
            segmentation_logits = 0.5 * (
                segmentation_logits + torch.flip(flipped_outputs["segmentation"], dims=(-1,))
            )
        binary_predictions = binary_prediction_from_logits(segmentation_logits, seg_threshold)
        valid = masks != IGNORE_ID
        pred_fg = (binary_predictions == 1) & valid
        target_fg = (masks == 1) & valid
        binary_intersection += int((pred_fg & target_fg).sum().item())
        binary_union += int((pred_fg | target_fg).sum().item())
        binary_correct_pixels += int(((binary_predictions == masks) & valid).sum().item())
        binary_valid_pixels += int(valid.sum().item())
        predicted_foreground_pixels += int(pred_fg.sum().item())
        target_foreground_pixels += int(target_fg.sum().item())

        oracle_semantic = torch.zeros_like(binary_predictions)
        oracle_ids = (class_ids.long() + 1).view(-1, 1, 1)
        oracle_semantic = torch.where(binary_predictions == 1, oracle_ids.expand_as(binary_predictions), oracle_semantic)
        valid_semantic = semantic_masks != IGNORE_ID
        for segmentation_id in torch.unique(semantic_masks[valid_semantic]).detach().cpu().tolist():
            segmentation_id = int(segmentation_id)
            if segmentation_id <= 0:
                continue
            pred_class = (oracle_semantic == segmentation_id) & valid_semantic
            target_class = (semantic_masks == segmentation_id) & valid_semantic
            oracle_intersections[segmentation_id] += float((pred_class & target_class).sum().item())
            oracle_unions[segmentation_id] += float((pred_class | target_class).sum().item())

        if segmentation_criterion is not None:
            running_segmentation_loss += float(segmentation_criterion(segmentation_logits, masks).item())
        if classification_criterion is not None:
            running_classification_loss += float(classification_criterion(classification_logits, class_ids).item())
        if segmentation_criterion is not None or classification_criterion is not None:
            loss_batches += 1
        class_tracker.update(classification_logits, class_ids)

        for item_index, image_name in enumerate(batch["image_name"]):
            height = int(batch["original_height"][item_index])
            width = int(batch["original_width"][item_index])
            mask, class_id = semantic_mask_from_binary_and_class_logits(
                segmentation_logits,
                classification_logits,
                item_index,
                height,
                width,
                seg_threshold,
            )
            submission_rows.append(
                {
                    "image": str(image_name),
                    "class_id": class_id,
                    "segmentation_rle": encode_mask_to_rle(mask),
                }
            )

    submission = pd.DataFrame(submission_rows, columns=["image", "class_id", "segmentation_rle"])
    solution = build_val_solution_frame(data_root, set(submission["image"].astype(str)))
    metrics = detailed_score(solution, submission)
    classification_metrics = class_tracker.compute()
    oracle_present = oracle_unions > 0
    oracle_miou = float(np.mean(oracle_intersections[oracle_present] / oracle_unions[oracle_present])) if oracle_present.any() else 0.0
    metrics["accuracy"] = classification_metrics["accuracy"]
    metrics["macro_accuracy_observed"] = classification_metrics["macro_accuracy"]
    metrics["binary_foreground_iou"] = binary_intersection / max(1, binary_union)
    metrics["binary_pixel_accuracy"] = binary_correct_pixels / max(1, binary_valid_pixels)
    metrics["oracle_semantic_miou"] = oracle_miou
    metrics["predicted_foreground_pct"] = 100.0 * predicted_foreground_pixels / max(1, binary_valid_pixels)
    metrics["target_foreground_pct"] = 100.0 * target_foreground_pixels / max(1, binary_valid_pixels)
    metrics["segmentation_loss"] = running_segmentation_loss / max(1, loss_batches)
    metrics["classification_loss"] = running_classification_loss / max(1, loss_batches)
    return {key: float(value) for key, value in metrics.items()}


def _optional_int(value: object) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value in {None, "", "None"}:
        return None
    return str(value)


def _optional_bool(value: object, default: bool) -> bool:
    if value in {None, "", "None"}:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}
