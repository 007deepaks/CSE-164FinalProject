"""PyTorch datasets for image-level classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.data.segmentation_dataset import DEFAULT_IMAGE_SIZE, image_to_tensor


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
    ) -> None:
        if split not in {"train_labeled", "val", "test"}:
            raise ValueError("split must be 'train_labeled', 'val', or 'test'")
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.samples = self._load_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_samples(self) -> list[ClassificationSample]:
        if self.split == "train_labeled":
            rows = _read_json(self.data_root / "metadata" / "train_labeled.json")
            return [
                ClassificationSample(
                    image_path=self.data_root / str(row["image"]),
                    image_name=Path(str(row["image"])).name,
                    class_id=int(row["class_id"]),
                )
                for row in rows
            ]

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
            image_tensor = image_to_tensor(image, self.image_size)
            original_size = image.size

        item: dict[str, object] = {
            "image": image_tensor,
            "image_name": sample.image_name,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }
        if sample.class_id is not None:
            item["class_id"] = int(sample.class_id)
        return item
