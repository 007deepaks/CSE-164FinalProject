#!/usr/bin/env python3
"""Kaggle-compatible metric for CSE 164 2026 CSV/RLE submissions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from kaggle_metric_utilities import ParticipantVisibleError
except ImportError:  # Local testing outside Kaggle.
    class ParticipantVisibleError(ValueError):
        pass


NUM_CLASSES = 300
IGNORE_ID = 1000


def encode_mask_ids(mask_ids: np.ndarray) -> str:
    """Encode a 2D id mask as row-major 1-indexed RLE triples.

    The encoded string is a space-separated sequence of:

        start length value start length value ...

    Only non-background pixels are stored. `start` is 1-indexed after row-major
    flattening, `length` is the run length, and `value` is the segmentation id.
    """
    flat = np.asarray(mask_ids, dtype=np.int64).reshape(-1)
    nonzero = flat != 0
    if not np.any(nonzero):
        return ""
    idx = np.flatnonzero(nonzero)
    values = flat[idx]

    run_break = np.ones(len(idx), dtype=bool)
    run_break[1:] = (idx[1:] != idx[:-1] + 1) | (values[1:] != values[:-1])
    starts = np.flatnonzero(run_break)
    ends = np.r_[starts[1:], len(idx)]

    parts: list[str] = []
    for start_pos, end_pos in zip(starts, ends):
        start = int(idx[start_pos]) + 1
        length = int(end_pos - start_pos)
        value = int(values[start_pos])
        parts.extend([str(start), str(length), str(value)])
    return " ".join(parts)


def _is_missing_rle(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def decode_rle_to_mask(
    rle: object,
    height: int,
    width: int,
    num_classes: int = NUM_CLASSES,
    allow_ignore: bool = False,
) -> np.ndarray:
    """Decode row-major RLE triples into a dense segmentation-id mask."""
    total = int(height) * int(width)
    mask = np.zeros(total, dtype=np.uint16)
    if _is_missing_rle(rle) or str(rle).strip() in {"", "0"}:
        return mask.reshape((int(height), int(width)))

    try:
        tokens = [int(tok) for tok in str(rle).split()]
    except ValueError as exc:
        raise ParticipantVisibleError("segmentation_rle must contain integer tokens only") from exc

    if len(tokens) % 3 != 0:
        raise ParticipantVisibleError("segmentation_rle must contain start length value triples")

    used = np.zeros(total, dtype=bool)
    for start, length, value in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if start < 1 or length < 1:
            raise ParticipantVisibleError("RLE starts and lengths must be positive")
        if value < 1 or (value > num_classes and not (allow_ignore and value == IGNORE_ID)):
            allowed = f"1..{num_classes}" + (f" or {IGNORE_ID}" if allow_ignore else "")
            raise ParticipantVisibleError(f"RLE values must be in {allowed}")
        begin = start - 1
        end = begin + length
        if end > total:
            raise ParticipantVisibleError("RLE run extends past the image size")
        if used[begin:end].any():
            raise ParticipantVisibleError("RLE runs must not overlap")
        used[begin:end] = True
        mask[begin:end] = value
    return mask.reshape((int(height), int(width)))


def _classification_metrics(pred: dict[str, int], gt: dict[str, int], num_classes: int) -> dict[str, float]:
    images = sorted(gt)
    correct = np.array([pred.get(image) == gt[image] for image in images], dtype=np.float64)
    accuracy = float(correct.mean()) if len(correct) else 0.0

    per_class = []
    for class_id in range(num_classes):
        class_images = [image for image in images if gt[image] == class_id]
        if class_images:
            per_class.append(float(np.mean([pred.get(image) == class_id for image in class_images])))
    macro_accuracy = float(np.mean(per_class)) if per_class else 0.0
    return {"accuracy": accuracy, "macro_accuracy": macro_accuracy}


def _confusion_matrix_for_masks(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    pred = np.where((pred >= 0) & (pred <= num_classes), pred, 0)
    valid = gt != IGNORE_ID
    valid &= gt >= 0
    valid &= gt <= num_classes
    labels = (num_classes + 1) * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    hist = np.bincount(labels, minlength=(num_classes + 1) ** 2)
    return hist.reshape(num_classes + 1, num_classes + 1)


def _iou_from_confusion(hist: np.ndarray, class_ids: list[int]) -> tuple[float, dict[int, float]]:
    per_class: dict[int, float] = {}
    for class_id in class_ids:
        tp = hist[class_id, class_id]
        fp = hist[:, class_id].sum() - tp
        fn = hist[class_id, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            per_class[class_id] = float(tp / denom)
    mean_iou = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean_iou, per_class


def _boundary_map(ids: np.ndarray) -> np.ndarray:
    valid = ids != IGNORE_ID
    boundary = np.zeros(ids.shape, dtype=bool)
    boundary[:-1, :] |= (ids[:-1, :] != ids[1:, :]) & valid[:-1, :] & valid[1:, :]
    boundary[1:, :] |= (ids[:-1, :] != ids[1:, :]) & valid[:-1, :] & valid[1:, :]
    boundary[:, :-1] |= (ids[:, :-1] != ids[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    boundary[:, 1:] |= (ids[:, :-1] != ids[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    return boundary


def _dilate_binary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    result = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            y_src_start = max(0, -dy)
            y_src_end = mask.shape[0] - max(0, dy)
            x_src_start = max(0, -dx)
            x_src_end = mask.shape[1] - max(0, dx)
            y_dst_start = max(0, dy)
            y_dst_end = mask.shape[0] - max(0, -dy)
            x_dst_start = max(0, dx)
            x_dst_end = mask.shape[1] - max(0, -dx)
            result[y_dst_start:y_dst_end, x_dst_start:x_dst_end] |= mask[
                y_src_start:y_src_end,
                x_src_start:x_src_end,
            ]
    return result


def _boundary_f_score(pred: np.ndarray, gt: np.ndarray, radius: int = 2) -> float:
    pred = pred.copy()
    pred[gt == IGNORE_ID] = IGNORE_ID
    pred_boundary = _boundary_map(pred)
    gt_boundary = _boundary_map(gt)
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0
    pred_match = pred_boundary & _dilate_binary(gt_boundary, radius)
    gt_match = gt_boundary & _dilate_binary(pred_boundary, radius)
    precision = pred_match.sum() / max(1, pred_boundary.sum())
    recall = gt_match.sum() / max(1, gt_boundary.sum())
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _validate_submission_frame(submission: pd.DataFrame, expected_images: set[str]) -> pd.DataFrame:
    required = {"image", "class_id", "segmentation_rle"}
    missing_columns = required - set(submission.columns)
    if missing_columns:
        raise ParticipantVisibleError(f"Submission is missing columns: {sorted(missing_columns)}")

    if submission["image"].duplicated().any():
        first = submission.loc[submission["image"].duplicated(), "image"].iloc[0]
        raise ParticipantVisibleError(f"Duplicate submission row for image {first}")

    actual_images = set(submission["image"].astype(str))
    missing = sorted(expected_images - actual_images)
    extra = sorted(actual_images - expected_images)
    if missing:
        raise ParticipantVisibleError(f"Missing predictions for {len(missing)} images; first: {missing[:5]}")
    if extra:
        raise ParticipantVisibleError(f"Extra predictions for {len(extra)} images; first: {extra[:5]}")

    result = submission.copy()
    result["image"] = result["image"].astype(str)
    try:
        result["class_id"] = result["class_id"].astype(int)
    except ValueError as exc:
        raise ParticipantVisibleError("class_id must be an integer") from exc
    bad = result[(result["class_id"] < 0) | (result["class_id"] >= NUM_CLASSES)]
    if not bad.empty:
        row = bad.iloc[0]
        raise ParticipantVisibleError(f"class_id out of range [0, {NUM_CLASSES - 1}] for {row['image']}")
    return result


def detailed_score(solution: pd.DataFrame, submission: pd.DataFrame) -> dict[str, float]:
    solution = solution.copy()
    solution["image"] = solution["image"].astype(str)
    expected_images = set(solution["image"])
    submission = _validate_submission_frame(submission, expected_images)

    sol = solution.set_index("image", drop=False)
    sub = submission.set_index("image", drop=False).loc[sol.index]

    pred_class = sub["class_id"].astype(int).to_dict()
    gt_class = sol["class_id"].astype(int).to_dict()
    classification = _classification_metrics(pred_class, gt_class, NUM_CLASSES)

    hist = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)
    boundary_scores = []
    gt_pixels = np.zeros(NUM_CLASSES + 1, dtype=np.int64)

    for image_name, gt_row in sol.iterrows():
        height = int(gt_row["height"])
        width = int(gt_row["width"])
        gt = decode_rle_to_mask(gt_row["segmentation_rle"], height, width, NUM_CLASSES, allow_ignore=True)
        pred = decode_rle_to_mask(sub.at[image_name, "segmentation_rle"], height, width, NUM_CLASSES)
        hist += _confusion_matrix_for_masks(pred, gt, NUM_CLASSES)
        boundary_scores.append(_boundary_f_score(pred, gt))
        for class_id in range(1, NUM_CLASSES + 1):
            gt_pixels[class_id] += int((gt == class_id).sum())

    mean_iou, per_class_iou = _iou_from_confusion(hist, list(range(1, NUM_CLASSES + 1)))
    present_classes = [cid for cid in range(1, NUM_CLASSES + 1) if gt_pixels[cid] > 0]
    rare_count = max(1, len(present_classes) // 4)
    rare_classes = sorted(present_classes, key=lambda cid: gt_pixels[cid])[:rare_count]
    rare_miou = float(np.mean([per_class_iou.get(cid, 0.0) for cid in rare_classes])) if rare_classes else 0.0
    boundary = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    segmentation_score = 0.70 * mean_iou + 0.20 * boundary + 0.10 * rare_miou

    automated_score = 0.70 * segmentation_score + 0.20 * classification["macro_accuracy"]
    return {
        "automated_score": automated_score,
        "segmentation_score": segmentation_score,
        "classification_macro_accuracy": classification["macro_accuracy"],
        "mean_iou": mean_iou,
        "boundary_f_score": boundary,
        "rare_class_miou": rare_miou,
    }


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "image") -> float:
    """Return the Kaggle leaderboard score. Higher is better.

    This is the automated 90% of the course score:

        70% segmentation + 20% classification

    The remaining 10% report/code reproducibility score is graded separately.
    """
    del row_id_column_name
    return detailed_score(solution, submission)["automated_score"]


def score_csv(solution_csv: str | Path, submission_csv: str | Path) -> dict[str, float]:
    return detailed_score(pd.read_csv(solution_csv), pd.read_csv(submission_csv))
