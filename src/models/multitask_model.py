"""Shared ConvNeXt multi-task model trained from scratch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.classification_model import ConvNeXtBlock, DownsampleLayer, LayerNorm2d, parse_depths


MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "tiny": {
        "base_channels": 96,
        "depths": (3, 3, 9, 3),
        "decoder_channels": 128,
    },
    "small": {
        "base_channels": 96,
        "depths": (3, 3, 27, 3),
        "decoder_channels": 192,
    },
}


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-style encoder that returns multi-scale feature maps."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,
        depths: tuple[int, int, int, int] = (3, 3, 27, 3),
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=4, stride=4),
            LayerNorm2d(self.channels[0]),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ConvNeXtBlock(self.channels[stage_index], mlp_ratio=mlp_ratio, drop_path=drop_path)
                        for _ in range(depth)
                    ]
                )
                for stage_index, depth in enumerate(depths)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                DownsampleLayer(self.channels[stage_index], self.channels[stage_index + 1])
                for stage_index in range(len(depths) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features: list[torch.Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if stage_index < len(self.downsamples):
                x = self.downsamples[stage_index](x)
        return features


class SegmentationDecoder(nn.Module):
    """Light FPN-style binary segmentation decoder."""

    def __init__(self, encoder_channels: list[int], decoder_channels: int = 192, num_segmentation_classes: int = 2) -> None:
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, kernel_size=1) for channels in encoder_channels]
        )
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(min(16, decoder_channels), decoder_channels),
                    nn.GELU(),
                )
                for _ in encoder_channels
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, decoder_channels // 2), decoder_channels // 2),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, num_segmentation_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        x = self.laterals[-1](features[-1])
        x = self.refine[-1](x)
        for feature_index in range(len(features) - 2, -1, -1):
            lateral = self.laterals[feature_index](features[feature_index])
            x = F.interpolate(x, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            x = self.refine[feature_index](x + lateral)
        x = self.head(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class ConvNeXtMultiTaskModel(nn.Module):
    """Shared ConvNeXt encoder with classification and binary segmentation heads."""

    def __init__(
        self,
        num_classes: int = 300,
        num_segmentation_classes: int = 2,
        base_channels: int = 96,
        depths: tuple[int, int, int, int] = (3, 3, 27, 3),
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        decoder_channels: int = 192,
    ) -> None:
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            base_channels=base_channels,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )
        encoder_channels = self.encoder.channels
        self.class_norm = nn.LayerNorm(encoder_channels[-1], eps=1e-6)
        self.class_head = nn.Linear(encoder_channels[-1], num_classes)
        self.segmentation_head = SegmentationDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_segmentation_classes=num_segmentation_classes,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        pooled = features[-1].mean(dim=(-2, -1))
        class_logits = self.class_head(self.class_norm(pooled))
        segmentation_logits = self.segmentation_head(features, output_size=x.shape[-2:])
        return {
            "classification": class_logits,
            "segmentation": segmentation_logits,
        }


def resolve_model_config(
    model_size: str = "small",
    base_channels: int | None = None,
    depths: str | tuple[int, int, int, int] | None = None,
    decoder_channels: int | None = None,
) -> dict[str, object]:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"model_size must be one of {sorted(MODEL_CONFIGS)}")
    config = dict(MODEL_CONFIGS[model_size])
    if base_channels is not None:
        config["base_channels"] = base_channels
    if depths is not None:
        config["depths"] = parse_depths(depths) if isinstance(depths, str) else depths
    if decoder_channels is not None:
        config["decoder_channels"] = decoder_channels
    return config


def build_multitask_model(
    model_size: str = "small",
    num_classes: int = 300,
    num_segmentation_classes: int = 2,
    base_channels: int | None = None,
    depths: str | tuple[int, int, int, int] | None = None,
    mlp_ratio: int = 4,
    drop_path: float = 0.0,
    decoder_channels: int | None = None,
) -> ConvNeXtMultiTaskModel:
    config = resolve_model_config(model_size, base_channels, depths, decoder_channels)
    return ConvNeXtMultiTaskModel(
        num_classes=num_classes,
        num_segmentation_classes=num_segmentation_classes,
        base_channels=int(config["base_channels"]),
        depths=config["depths"],  # type: ignore[arg-type]
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        decoder_channels=int(config["decoder_channels"]),
    )
