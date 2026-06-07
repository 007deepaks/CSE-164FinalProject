"""PyTorch datasets for image-level classification."""

from __future__ import annotations

import random
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data import Dataset

from src.data.segmentation_dataset import DEFAULT_IMAGE_SIZE, augment_image_to_tensor, image_to_tensor


@dataclass(frozen=True)
class ClassificationSample:
    image_path: Path
    image_name: str
    class_id: int | None = None


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
    ) -> None:
        if split not in {"train_labeled", "train_combined", "val", "test"}:
            raise ValueError("split must be 'train_labeled', 'train_combined', 'val', or 'test'")
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.random_crop = random_crop
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
                samples.extend(
                    ClassificationSample(
                        image_path=self.data_root / str(row["image"]),
                        image_name=Path(str(row["image"])).name,
                        class_id=int(row["class_id"]),
                    )
                    for row in seg_rows
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
            original_size = image.size
            image_tensor = (
                augment_image_to_tensor(image, self.image_size, self.random_crop)
                if self.augment
                else image_to_tensor(image, self.image_size)
            )

        item: dict[str, object] = {
            "image": image_tensor,
            "image_name": sample.image_name,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }
        if sample.class_id is not None:
            item["class_id"] = int(sample.class_id)
        return item


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
