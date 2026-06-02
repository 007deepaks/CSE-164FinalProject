"""PyTorch datasets for the CSE 164 segmentation baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.masks import decode_rgb_mask

DEFAULT_IMAGE_SIZE = 256
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path | None
    image_name: str


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def image_to_tensor(image: Image.Image, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Resize an RGB image and convert it to a normalized CHW float tensor."""
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def mask_to_tensor(mask_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Decode and resize a mask, preserving integer ids and ignore label 1000."""
    mask = decode_rgb_mask(mask_path).astype(np.int32)
    mask_image = Image.fromarray(mask, mode="I")
    mask_image = mask_image.resize((image_size, image_size), Image.Resampling.NEAREST)
    return torch.from_numpy(np.asarray(mask_image, dtype=np.int64))


class SegmentationDataset(Dataset[dict[str, object]]):
    """Dataset for train_seg and val segmentation splits."""

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        split: str = "train_seg",
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"train_seg", "val"}:
            raise ValueError("split must be 'train_seg' or 'val'")
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.samples = self._load_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_samples(self) -> list[SegmentationSample]:
        if self.split == "train_seg":
            rows = _read_json(self.data_root / "metadata" / "train_seg.json")
            return [
                SegmentationSample(
                    image_path=self.data_root / str(row["image"]),
                    mask_path=self.data_root / str(row["mask"]),
                    image_name=Path(str(row["image"])).name,
                )
                for row in rows
            ]

        image_dir = self.data_root / "val" / "images"
        mask_dir = self.data_root / "val" / "masks"
        return [
            SegmentationSample(
                image_path=image_path,
                mask_path=mask_dir / f"{image_path.stem}.png",
                image_name=image_path.name,
            )
            for image_path in sorted(image_dir.glob("*.JPEG"))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image_tensor = image_to_tensor(image, self.image_size)
            original_size = image.size
        if sample.mask_path is None:
            raise ValueError("SegmentationDataset requires masks")
        mask_tensor = mask_to_tensor(sample.mask_path, self.image_size)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": sample.image_name,
            "original_size": original_size,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }


class TestImageDataset(Dataset[dict[str, object]]):
    """Dataset for test images without masks."""

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_samples: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.image_paths = sorted((self.data_root / "test" / "images").glob("*.JPEG"))
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            original_size = image.size
            image_tensor = image_to_tensor(image, self.image_size)
        return {
            "image": image_tensor,
            "image_name": image_path.name,
            "original_size": original_size,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }
