"""Classification metrics."""

from __future__ import annotations

import numpy as np
import torch


class ClassificationMetricTracker:
    """Track top-1 accuracy and macro accuracy over observed classes."""

    def __init__(self, num_classes: int = 300) -> None:
        self.num_classes = num_classes
        self.correct = 0
        self.total = 0
        self.per_class_correct = np.zeros(num_classes, dtype=np.int64)
        self.per_class_total = np.zeros(num_classes, dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = torch.argmax(logits.detach(), dim=1)
        pred = prediction.cpu().numpy().astype(np.int64)
        gt = target.detach().cpu().numpy().astype(np.int64)
        self.correct += int((pred == gt).sum())
        self.total += int(len(gt))
        for class_id in range(self.num_classes):
            mask = gt == class_id
            if mask.any():
                self.per_class_total[class_id] += int(mask.sum())
                self.per_class_correct[class_id] += int((pred[mask] == class_id).sum())

    def compute(self) -> dict[str, float]:
        accuracy = self.correct / max(1, self.total)
        present = self.per_class_total > 0
        per_class_accuracy = np.divide(
            self.per_class_correct[present],
            self.per_class_total[present],
            out=np.zeros(int(present.sum()), dtype=np.float64),
            where=self.per_class_total[present] > 0,
        )
        macro_accuracy = float(per_class_accuracy.mean()) if len(per_class_accuracy) else 0.0
        return {
            "accuracy": float(accuracy),
            "macro_accuracy": macro_accuracy,
        }
