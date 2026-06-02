"""Small ConvNeXt-style image classifier trained from scratch."""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with depthwise conv, MLP, GELU, and residual."""

    def __init__(self, channels: int, mlp_ratio: int = 4, drop_path: float = 0.0) -> None:
        super().__init__()
        hidden = channels * mlp_ratio
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pointwise1 = nn.Linear(channels, hidden)
        self.activation = nn.GELU()
        self.pointwise2 = nn.Linear(hidden, channels)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvNeXtClassifier(nn.Module):
    """Compact ConvNeXt-style classifier for 300 CSE164 classes."""

    def __init__(
        self,
        num_classes: int = 300,
        in_channels: int = 3,
        base_channels: int = 48,
        depths: tuple[int, int, int, int] = (2, 2, 4, 2),
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=4, stride=4),
            LayerNorm2d(channels[0]),
        )
        stages: list[nn.Module] = []
        for stage_index, depth in enumerate(depths):
            blocks = [
                ConvNeXtBlock(channels[stage_index], mlp_ratio=mlp_ratio, drop_path=drop_path)
                for _ in range(depth)
            ]
            stages.append(nn.Sequential(*blocks))
            if stage_index < len(depths) - 1:
                stages.append(DownsampleLayer(channels[stage_index], channels[stage_index + 1]))
        self.features = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(channels[-1], eps=1e-6)
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = x.mean(dim=(-2, -1))
        x = self.norm(x)
        return self.head(x)


def parse_depths(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4 or any(part < 1 for part in parts):
        raise ValueError("depths must contain four positive integers, e.g. '2,2,4,2'")
    return parts


def build_classification_model(
    num_classes: int = 300,
    base_channels: int = 48,
    depths: tuple[int, int, int, int] = (2, 2, 4, 2),
    mlp_ratio: int = 4,
    drop_path: float = 0.0,
) -> ConvNeXtClassifier:
    return ConvNeXtClassifier(
        num_classes=num_classes,
        base_channels=base_channels,
        depths=depths,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
    )
