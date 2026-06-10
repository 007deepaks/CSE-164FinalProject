"""Shape checks for the ResNet-50 multi-task model."""

from __future__ import annotations

import torch

from src.models.multitaskResnet50 import build_resnet50_multitask_model


def test_resnet50_multitask_shapes() -> None:
    model = build_resnet50_multitask_model(num_segmentation_classes=1)
    model.eval()
    x = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        out = model(x, seg=True)
        assert out["classification"].shape == (2, 300)
        assert out["segmentation"].shape == (2, 1, 384, 384)

        out = model(x, seg=False)
        assert out["classification"].shape == (2, 300)
        assert out["segmentation"] is None


if __name__ == "__main__":
    test_resnet50_multitask_shapes()
    print("ResNet-50 multi-task shape checks passed")
