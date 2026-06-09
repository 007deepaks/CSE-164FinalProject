"""PyTorch datasets for image-level classification."""

from __future__ import annotations

import random
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Sampler
from torch.utils.data import Dataset

from src.data.segmentation_dataset import DEFAULT_IMAGE_SIZE, _normalize_image, augment_image_to_tensor, image_to_tensor
from src.utils.masks import IGNORE_ID, decode_rgb_mask


@dataclass(frozen=True)
class ClassificationSample:
    image_path: Path
    image_name: str
    class_id: int | None = None
    mask_path: Path | None = None
    crop_mode: str = "full"


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class ClassificationDataset(Dataset[dict[str, object]]):
    """Dataset for train_labeled, val, and test image classification."""

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        split: str = "train_labeled",
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_samples: int | None = None,
        augment: bool = False,
        random_crop: bool = True,
        augment_policy: str = "basic",
        include_seg_crops: bool = False,
        crop_padding: float = 0.15,
    ) -> None:
        if split not in {"train_labeled", "train_combined", "val", "test"}:
            raise ValueError("split must be 'train_labeled', 'train_combined', 'val', or 'test'")
        if augment_policy not in {"basic", "strong"}:
            raise ValueError("augment_policy must be 'basic' or 'strong'")
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.random_crop = random_crop
        self.augment_policy = augment_policy
        self.include_seg_crops = include_seg_crops
        self.crop_padding = crop_padding
        self.samples = self._load_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_samples(self) -> list[ClassificationSample]:
        if self.split in {"train_labeled", "train_combined"}:
            labeled_rows = _read_json(self.data_root / "metadata" / "train_labeled.json")
            samples = [
                ClassificationSample(
                    image_path=self.data_root / str(row["image"]),
                    image_name=Path(str(row["image"])).name,
                    class_id=int(row["class_id"]),
                )
                for row in labeled_rows
            ]
            if self.split == "train_combined":
                seg_rows = _read_json(self.data_root / "metadata" / "train_seg.json")
                seg_samples = [
                    ClassificationSample(
                        image_path=self.data_root / str(row["image"]),
                        image_name=Path(str(row["image"])).name,
                        class_id=int(row["class_id"]),
                        mask_path=self.data_root / str(row["mask"]),
                    )
                    for row in seg_rows
                ]
                samples.extend(seg_samples)
                if self.include_seg_crops:
                    samples.extend(
                        ClassificationSample(
                            image_path=sample.image_path,
                            image_name=f"{sample.image_path.stem}_crop{sample.image_path.suffix}",
                            class_id=sample.class_id,
                            mask_path=sample.mask_path,
                            crop_mode="mask_crop",
                        )
                        for sample in seg_samples
                    )
            return samples

        if self.split == "val":
            rows = _read_json(self.data_root / "val" / "classification.json")
            return [
                ClassificationSample(
                    image_path=self.data_root / "val" / "images" / str(row["image"]),
                    image_name=str(row["image"]),
                    class_id=int(row["class_id"]),
                )
                for row in rows
            ]

        image_paths = sorted((self.data_root / "test" / "images").glob("*.JPEG"))
        return [
            ClassificationSample(
                image_path=image_path,
                image_name=image_path.name,
                class_id=None,
            )
            for image_path in image_paths
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            if sample.crop_mode == "mask_crop":
                image = crop_image_from_mask(image, sample.mask_path, self.crop_padding)
            original_size = image.size
            if self.augment and self.augment_policy == "strong":
                image_tensor = strong_supervised_image_to_tensor(image, self.image_size, self.random_crop)
            elif self.augment:
                image_tensor = augment_image_to_tensor(image, self.image_size, self.random_crop)
            else:
                image_tensor = image_to_tensor(image, self.image_size)

        item: dict[str, object] = {
            "image": image_tensor,
            "image_name": sample.image_name,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }
        if sample.class_id is not None:
            item["class_id"] = int(sample.class_id)
        return item


def strong_supervised_image_to_tensor(
    image: Image.Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    random_crop: bool = True,
) -> torch.Tensor:
    """Controlled stronger supervised augmentation for tiny 300-way classifier data."""
    image = image.convert("RGB")
    if random_crop:
        image = _random_square_crop(image)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.70, 1.30))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.70, 1.35))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.30))
    if random.random() < 0.12:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.7)))
    if random.random() < 0.08:
        image = ImageOps.grayscale(image).convert("RGB")
    if random.random() < 0.06:
        image = ImageOps.solarize(image, threshold=random.randint(112, 192))
    if random.random() < 0.06:
        image = ImageOps.posterize(image, bits=random.randint(4, 6))
    return _normalize_image(image)


class UnlabeledFixMatchDataset(Dataset[dict[str, object]]):
    """Return weak and strong views of train_unlabeled images for classifier FixMatch."""

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_samples: int | None = None,
        random_crop: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.random_crop = random_crop
        image_root = self.data_root / "train_unlabeled"
        candidates = sorted(image_root.rglob("*.JPEG"))
        if not candidates:
            candidates = sorted(image_root.rglob("*.jpg")) + sorted(image_root.rglob("*.png"))
        self.image_paths = candidates
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
        if not self.image_paths:
            raise ValueError(f"No unlabeled images found under {image_root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            weak = weak_fixmatch_image_to_tensor(image, self.image_size, self.random_crop)
            strong = strong_fixmatch_image_to_tensor(image, self.image_size, self.random_crop)
        return {
            "weak_image": weak,
            "strong_image": strong,
            "image_name": image_path.name,
        }


def weak_fixmatch_image_to_tensor(
    image: Image.Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    random_crop: bool = False,
) -> torch.Tensor:
    """Weak unlabeled view: resize plus optional crop and horizontal flip."""
    image = image.convert("RGB")
    if random_crop:
        image = augment_image_to_tensor(image, image_size, random_crop=True)
        return image
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return _normalize_image(image)


def strong_fixmatch_image_to_tensor(
    image: Image.Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    random_crop: bool = False,
) -> torch.Tensor:
    """Strong unlabeled view with heavier PIL transforms and tensor cutout."""
    image = image.convert("RGB")
    if random_crop:
        image = _random_square_crop(image)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = _strong_jitter_image(image)
    tensor = _normalize_image(image)
    if random.random() < 0.65:
        tensor = _cutout_tensor(tensor, max_fraction=0.35)
    return tensor


def _random_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_size = random.randint(int(0.75 * min(width, height)), min(width, height))
    left = random.randint(0, max(0, width - crop_size))
    top = random.randint(0, max(0, height - crop_size))
    return image.crop((left, top, left + crop_size, top + crop_size))


def _strong_jitter_image(image: Image.Image) -> Image.Image:
    operations = [
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.45, 1.55)),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.45, 1.65)),
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.35, 1.65)),
        lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.3, 2.0)),
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2))),
        lambda img: ImageOps.solarize(img, threshold=random.randint(96, 192)),
        lambda img: ImageOps.posterize(img, bits=random.randint(4, 6)),
    ]
    for operation in random.sample(operations, k=random.randint(2, 4)):
        image = operation(image)
    if random.random() < 0.15:
        image = ImageOps.grayscale(image).convert("RGB")
    return image


def _cutout_tensor(tensor: torch.Tensor, max_fraction: float = 0.35) -> torch.Tensor:
    _, height, width = tensor.shape
    erase_height = random.randint(1, max(1, int(height * max_fraction)))
    erase_width = random.randint(1, max(1, int(width * max_fraction)))
    top = random.randint(0, max(0, height - erase_height))
    left = random.randint(0, max(0, width - erase_width))
    tensor = tensor.clone()
    tensor[:, top : top + erase_height, left : left + erase_width] = 0.0
    return tensor


def crop_image_from_mask(image: Image.Image, mask_path: Path | None, padding_fraction: float) -> Image.Image:
    """Crop an image around non-background mask pixels, with proportional padding."""
    if mask_path is None:
        return image
    mask = decode_rgb_mask(mask_path)
    foreground = (mask > 0) & (mask != IGNORE_ID)
    if not foreground.any():
        return image
    ys, xs = np.where(foreground)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    box_width = right - left
    box_height = bottom - top
    pad = int(round(max(box_width, box_height) * max(0.0, padding_fraction)))
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    return image.crop((left, top, right, bottom))


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Build batches by sampling class ids uniformly, then an image within each class."""

    def __init__(
        self,
        samples: list[ClassificationSample],
        batch_size: int,
        batches_per_epoch: int | None = None,
        seed: int = 164,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch or max(1, len(samples) // batch_size)
        self.seed = seed
        self.indices_by_class: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            if sample.class_id is None:
                continue
            self.indices_by_class[int(sample.class_id)].append(index)
        if not self.indices_by_class:
            raise ValueError("BalancedClassBatchSampler requires class labels")
        self.class_ids = sorted(self.indices_by_class)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for _ in range(self.batch_size):
                class_id = rng.choice(self.class_ids)
                batch.append(rng.choice(self.indices_by_class[class_id]))
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
