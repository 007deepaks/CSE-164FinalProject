"""ResNet/ResNeXt multi-task models trained from scratch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    """ResNet-family bottleneck block with optional grouped 3x3 convolution."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        width = int(channels * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class ResNet50Encoder(nn.Module):
    """ResNet-50/ResNeXt-50 encoder returning C1-C4 feature maps."""

    def __init__(self, in_channels: int = 3, groups: int = 1, base_width: int = 64) -> None:
        super().__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self.channels = [256, 512, 1024, 2048]
        self._init_weights()

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        out_channels = channels * Bottleneck.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers: list[nn.Module] = [
            Bottleneck(
                self.inplanes,
                channels,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
            )
        ]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, channels, groups=self.groups, base_width=self.base_width))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetUNetDecoder(nn.Module):
    """U-Net decoder for ResNet C1-C4 skip features."""

    def __init__(self, encoder_channels: list[int], num_segmentation_classes: int = 1) -> None:
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.decode3 = ConvBlock(c4 + c3, 512)
        self.decode2 = ConvBlock(512 + c2, 256)
        self.decode1 = ConvBlock(256 + c1, 128)
        self.final = nn.Sequential(
            ConvBlock(128, 64),
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


class ResNet50MultiTaskModel(nn.Module):
    """Shared ResNet-family encoder with binary segmentation and mask-guided classification."""

    def __init__(
        self,
        num_classes: int = 300,
        num_segmentation_classes: int = 1,
        dropout: float = 0.2,
        groups: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder(groups=groups, base_width=base_width)
        self.segmentation_head = ResNetUNetDecoder(
            self.encoder.channels,
            num_segmentation_classes=num_segmentation_classes,
        )
        classifier_channels = self.encoder.channels[-1] * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_channels, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(classifier_channels, num_classes),
        )

    def _classification_logits(
        self,
        deepest_feature: torch.Tensor,
        segmentation_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        global_feat = deepest_feature.mean(dim=(-2, -1))
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
        return self.classifier(torch.cat([global_feat, foreground_feat], dim=1))

    def forward(self, x: torch.Tensor, seg: bool = True) -> dict[str, torch.Tensor | None]:
        features = self.encoder(x)
        segmentation_logits = self.segmentation_head(features, output_size=x.shape[-2:]) if seg else None
        class_logits = self._classification_logits(features[-1], segmentation_logits)
        return {
            "classification": class_logits,
            "segmentation": segmentation_logits,
        }


def build_resnet50_multitask_model(
    num_classes: int = 300,
    num_segmentation_classes: int = 1,
    dropout: float = 0.2,
    architecture: str = "resnet50",
) -> ResNet50MultiTaskModel:
    if architecture == "resnet50":
        groups = 1
        base_width = 64
    elif architecture == "resnext50_32x4d":
        groups = 32
        base_width = 4
    else:
        raise ValueError("architecture must be 'resnet50' or 'resnext50_32x4d'")
    return ResNet50MultiTaskModel(
        num_classes=num_classes,
        num_segmentation_classes=num_segmentation_classes,
        dropout=dropout,
        groups=groups,
        base_width=base_width,
    )
