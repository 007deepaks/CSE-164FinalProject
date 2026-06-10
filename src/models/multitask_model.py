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
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
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


class FPNSegmentationDecoder(nn.Module):
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSegmentationDecoder(nn.Module):
    """U-Net-style decoder with ConvNeXt skip connections."""

    def __init__(self, encoder_channels: list[int], num_segmentation_classes: int = 1) -> None:
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.decode3 = ConvBlock(c4 + c3, 256)
        self.decode2 = ConvBlock(256 + c2, 128)
        self.decode1 = ConvBlock(128 + c1, 64)
        self.final = nn.Sequential(
            ConvBlock(64, 64),
            nn.Conv2d(64, num_segmentation_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        x = F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode3(torch.cat([x, c3], dim=1))
        x = F.interpolate(x, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode2(torch.cat([x, c2], dim=1))
        x = F.interpolate(x, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode1(torch.cat([x, c1], dim=1))
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.final(x)


class ConvNeXtMultiTaskModel(nn.Module):
    """Shared ConvNeXt encoder with classification and binary segmentation heads."""

    def __init__(
        self,
        num_classes: int = 300,
        num_segmentation_classes: int = 1,
        base_channels: int = 96,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        decoder_channels: int = 192,
        decoder_type: str = "unet",
        mask_guided_classifier: bool = True,
    ) -> None:
        super().__init__()
        if decoder_type not in {"unet", "fpn"}:
            raise ValueError("decoder_type must be 'unet' or 'fpn'")
        self.encoder = ConvNeXtEncoder(
            base_channels=base_channels,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )
        encoder_channels = self.encoder.channels
        self.mask_guided_classifier = mask_guided_classifier
        classifier_channels = encoder_channels[-1] * 2 if mask_guided_classifier else encoder_channels[-1]
        self.class_norm = nn.LayerNorm(classifier_channels, eps=1e-6)
        self.class_dropout = nn.Dropout(0.1)
        self.class_head = nn.Linear(classifier_channels, num_classes)
        if decoder_type == "unet":
            self.segmentation_head = UNetSegmentationDecoder(
                encoder_channels=encoder_channels,
                num_segmentation_classes=num_segmentation_classes,
            )
        else:
            self.segmentation_head = FPNSegmentationDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                num_segmentation_classes=num_segmentation_classes,
            )

    def _classification_logits(
        self,
        deepest_feature: torch.Tensor,
        segmentation_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        global_feat = deepest_feature.mean(dim=(-2, -1))
        if self.mask_guided_classifier:
            if segmentation_logits is None:
                foreground_feat = global_feat
            else:
                if segmentation_logits.shape[1] == 1:
                    foreground_prob = torch.sigmoid(segmentation_logits)
                else:
                    foreground_prob = torch.softmax(segmentation_logits, dim=1)[:, 1:2]
                foreground_prob = F.interpolate(
                    foreground_prob.detach(),
                    size=deepest_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                weights = foreground_prob / (foreground_prob.sum(dim=(2, 3), keepdim=True) + 1e-6)
                foreground_feat = (deepest_feature * weights).sum(dim=(2, 3))
            pooled = torch.cat([global_feat, foreground_feat], dim=1)
        else:
            pooled = global_feat
        return self.class_head(self.class_dropout(self.class_norm(pooled)))

    def forward(self, x: torch.Tensor, seg: bool = True) -> dict[str, torch.Tensor | None]:
        features = self.encoder(x)
        segmentation_logits = self.segmentation_head(features, output_size=x.shape[-2:]) if seg else None
        class_logits = self._classification_logits(features[-1], segmentation_logits)
        return {
            "classification": class_logits,
            "segmentation": segmentation_logits,
        }


def resolve_model_config(
    model_size: str = "tiny",
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
    model_size: str = "tiny",
    num_classes: int = 300,
    num_segmentation_classes: int = 1,
    base_channels: int | None = None,
    depths: str | tuple[int, int, int, int] | None = None,
    mlp_ratio: int = 4,
    drop_path: float = 0.0,
    decoder_channels: int | None = None,
    decoder_type: str = "unet",
    mask_guided_classifier: bool = True,
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
        decoder_type=decoder_type,
        mask_guided_classifier=mask_guided_classifier,
    )
