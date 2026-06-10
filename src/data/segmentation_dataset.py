"""PyTorch datasets for the CSE 164 segmentation baseline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
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
    class_id: int | None = None


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_image(image: Image.Image) -> torch.Tensor:
    """Convert an already-sized RGB image to a normalized CHW float tensor."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def image_to_tensor(image: Image.Image, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Resize an RGB image and convert it to a normalized CHW float tensor."""
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return _normalize_image(image)


def mask_to_tensor(mask_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Decode and resize a mask, preserving integer ids and ignore label 1000."""
    mask = decode_rgb_mask(mask_path).astype(np.int32)
    mask_image = Image.fromarray(mask, mode="I")
    mask_image = mask_image.resize((image_size, image_size), Image.Resampling.NEAREST)
    return torch.from_numpy(np.asarray(mask_image, dtype=np.int64))


def _mask_image_to_tensor(mask_image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(mask_image, dtype=np.int64))


def _random_resized_crop_params(width: int, height: int) -> tuple[int, int, int, int]:
    area = width * height
    for _ in range(10):
        target_area = random.uniform(0.55, 1.0) * area
        aspect_ratio = random.uniform(0.75, 1.3333333333)
        crop_width = int(round((target_area * aspect_ratio) ** 0.5))
        crop_height = int(round((target_area / aspect_ratio) ** 0.5))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            return left, top, crop_width, crop_height

    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return left, top, crop_size, crop_size


def _jitter_image(image: Image.Image) -> Image.Image:
    brightness = random.uniform(0.75, 1.25)
    contrast = random.uniform(0.75, 1.25)
    color = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    if random.random() < 0.10:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
    if random.random() < 0.10:
        array = np.asarray(image, dtype=np.int16)
        noise = np.random.normal(0, random.uniform(2.0, 8.0), size=array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
    return image


def _resize_random_crop_pair(
    image: Image.Image,
    mask_image: Image.Image,
    image_size: int,
) -> tuple[Image.Image, Image.Image]:
    resize_size = int(round(image_size * 1.14))
    image = image.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
    mask_image = mask_image.resize((resize_size, resize_size), Image.Resampling.NEAREST)

    if random.random() < 0.75:
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.90, 1.10)
        translate_x = random.uniform(-0.05, 0.05) * resize_size
        translate_y = random.uniform(-0.05, 0.05) * resize_size
        center = resize_size * 0.5
        radians = np.deg2rad(angle)
        cos_value = np.cos(radians) / scale
        sin_value = np.sin(radians) / scale
        a = cos_value
        b = sin_value
        d = -sin_value
        e = cos_value
        c = center - a * center - b * center - translate_x
        f = center - d * center - e * center - translate_y
        affine = (a, b, c, d, e, f)
        image = image.transform(
            (resize_size, resize_size),
            Image.Transform.AFFINE,
            affine,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        mask_image = mask_image.transform(
            (resize_size, resize_size),
            Image.Transform.AFFINE,
            affine,
            resample=Image.Resampling.NEAREST,
            fillcolor=1000,
        )

    left = random.randint(0, resize_size - image_size)
    top = random.randint(0, resize_size - image_size)
    crop_box = (left, top, left + image_size, top + image_size)
    return image.crop(crop_box), mask_image.crop(crop_box)


def augment_image_to_tensor(
    image: Image.Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    random_crop: bool = True,
) -> torch.Tensor:
    """Apply image-only training augmentation and return a normalized tensor."""
    image = image.convert("RGB")
    if random_crop:
        width, height = image.size
        left, top, crop_width, crop_height = _random_resized_crop_params(width, height)
        image = image.crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = _jitter_image(image)
    return _normalize_image(image)


def augment_image_and_mask_to_tensors(
    image: Image.Image,
    mask: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    random_crop: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply synchronized crop/flip to image and mask, with jitter on image only."""
    image = image.convert("RGB")
    mask_image = Image.fromarray(mask.astype(np.int32), mode="I")
    if random_crop:
        image, mask_image = _resize_random_crop_pair(image, mask_image, image_size)
    else:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        mask_image = mask_image.resize((image_size, image_size), Image.Resampling.NEAREST)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = _jitter_image(image)
    return _normalize_image(image), _mask_image_to_tensor(mask_image)


def semantic_to_binary_mask(semantic_mask: torch.Tensor) -> torch.Tensor:
    """Convert ids 1..300 to foreground=1 while preserving ignore=1000."""
    binary = torch.zeros_like(semantic_mask)
    binary[(semantic_mask > 0) & (semantic_mask != 1000)] = 1
    binary[semantic_mask == 1000] = 1000
    return binary


class SegmentationDataset(Dataset[dict[str, object]]):
    """Dataset for train_seg and val segmentation splits."""

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        split: str = "train_seg",
        image_size: int = DEFAULT_IMAGE_SIZE,
        target_mode: str = "binary",
        max_samples: int | None = None,
        augment: bool = False,
        random_crop: bool = True,
    ) -> None:
        if split not in {"train_seg", "val"}:
            raise ValueError("split must be 'train_seg' or 'val'")
        if target_mode not in {"binary", "semantic"}:
            raise ValueError("target_mode must be 'binary' or 'semantic'")
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.target_mode = target_mode
        self.augment = augment
        self.random_crop = random_crop
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
                    class_id=int(row["class_id"]),
                )
                for row in rows
            ]

        image_dir = self.data_root / "val" / "images"
        mask_dir = self.data_root / "val" / "masks"
        class_rows = _read_json(self.data_root / "val" / "classification.json")
        class_by_image = {str(row["image"]): int(row["class_id"]) for row in class_rows}
        return [
            SegmentationSample(
                image_path=image_path,
                mask_path=mask_dir / f"{image_path.stem}.png",
                image_name=image_path.name,
                class_id=class_by_image[image_path.name],
            )
            for image_path in sorted(image_dir.glob("*.JPEG"))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            original_size = image.size
        if sample.mask_path is None:
            raise ValueError("SegmentationDataset requires masks")
        semantic_mask_array = decode_rgb_mask(sample.mask_path).astype(np.int32)
        if self.augment:
            image_tensor, semantic_mask = augment_image_and_mask_to_tensors(
                image,
                semantic_mask_array,
                self.image_size,
                self.random_crop,
            )
        else:
            image_tensor = image_to_tensor(image, self.image_size)
            mask_image = Image.fromarray(semantic_mask_array, mode="I")
            mask_image = mask_image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            semantic_mask = _mask_image_to_tensor(mask_image)
        mask_tensor = semantic_to_binary_mask(semantic_mask) if self.target_mode == "binary" else semantic_mask
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "semantic_mask": semantic_mask,
            "class_id": int(sample.class_id) if sample.class_id is not None else -1,
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
