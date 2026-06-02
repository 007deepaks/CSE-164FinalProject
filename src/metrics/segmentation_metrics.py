"""Simple segmentation metrics for the first baseline."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.masks import IGNORE_ID, NUM_CLASSES


class SegmentationMetricTracker:
    """Accumulate pixel accuracy and foreground mean IoU while ignoring 1000."""

    def __init__(self, num_classes: int = NUM_CLASSES + 1, ignore_index: int = IGNORE_ID) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.correct_pixels = 0
        self.valid_pixels = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = torch.argmax(logits.detach(), dim=1)
        self.update_predictions(prediction, target)

    def update_predictions(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        pred = prediction.detach().cpu().numpy().astype(np.int64).reshape(-1)
        gt = target.detach().cpu().numpy().astype(np.int64).reshape(-1)
        valid = gt != self.ignore_index
        valid &= gt >= 0
        valid &= gt < self.num_classes
        pred = np.where((pred >= 0) & (pred < self.num_classes), pred, 0)
        pred = pred[valid]
        gt = gt[valid]
        self.correct_pixels += int((pred == gt).sum())
        self.valid_pixels += int(valid.sum())
        labels = self.num_classes * gt + pred
        hist = np.bincount(labels, minlength=self.num_classes**2)
        self.confusion += hist.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        pixel_accuracy = self.correct_pixels / max(1, self.valid_pixels)
        ious: list[float] = []
        for class_id in range(1, self.num_classes):
            tp = self.confusion[class_id, class_id]
            fp = self.confusion[:, class_id].sum() - tp
            fn = self.confusion[class_id, :].sum() - tp
            denom = tp + fp + fn
            if denom > 0:
                ious.append(float(tp / denom))
        return {
            "pixel_accuracy": float(pixel_accuracy),
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
        }
