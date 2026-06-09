"""Classification metrics."""

from __future__ import annotations

import numpy as np
import torch


class ClassificationMetricTracker:
    """Track top-1 accuracy and macro accuracy over observed classes."""

    def __init__(self, num_classes: int = 300) -> None:
        self.num_classes = num_classes
        self.correct = 0
        self.top5_correct = 0
        self.total = 0
        self.confidence_sum = 0.0
        self.per_class_correct = np.zeros(num_classes, dtype=np.int64)
        self.per_class_total = np.zeros(num_classes, dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        detached_logits = logits.detach()
        probabilities = torch.softmax(detached_logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        topk = torch.topk(detached_logits, k=min(5, detached_logits.shape[1]), dim=1).indices
        pred = prediction.cpu().numpy().astype(np.int64)
        gt = target.detach().cpu().numpy().astype(np.int64)
        self.correct += int((pred == gt).sum())
        self.top5_correct += int((topk == target.detach().view(-1, 1)).any(dim=1).sum().item())
        self.total += int(len(gt))
        self.confidence_sum += float(confidence.sum().cpu().item())
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
            "top5_accuracy": float(self.top5_correct / max(1, self.total)),
            "macro_accuracy": macro_accuracy,
            "mean_confidence": float(self.confidence_sum / max(1, self.total)),
        }
