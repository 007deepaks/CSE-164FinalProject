"""Inspect the Kaggle dataset layout, metadata, images, and masks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.masks import IGNORE_ID, NUM_CLASSES, decode_rgb_mask, validate_mask_ids

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = {
    "train_labeled": ("images",),
    "train_seg": ("images", "masks"),
    "train_unlabeled": ("images",),
    "val": ("images", "masks"),
    "test": ("images",),
}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_files(path: Path, suffixes: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in suffixes)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def summarize_sizes(paths: list[Path], limit: int) -> Counter[tuple[int, int]]:
    counter: Counter[tuple[int, int]] = Counter()
    for path in paths[:limit]:
        counter[image_size(path)] += 1
    return counter


def print_metadata_summary(data_root: Path) -> None:
    metadata_dir = data_root / "metadata"
    print("\nMetadata")
    for name in ("class_map.json", "train_labeled.json", "train_seg.json"):
        path = metadata_dir / name
        if not path.exists():
            print(f"  MISSING {path}")
            continue
        data = load_json(path)
        count = len(data) if hasattr(data, "__len__") else "unknown"
        print(f"  {name}: {type(data).__name__}, entries={count}")
        if isinstance(data, list) and data:
            print(f"    first: {data[0]}")

    val_classification = data_root / "val" / "classification.json"
    if val_classification.exists():
        rows = load_json(val_classification)
        print(f"  val/classification.json: entries={len(rows)}")


def print_split_counts(data_root: Path, size_sample: int) -> dict[str, list[Path]]:
    print("\nSplit counts and sampled image sizes")
    split_images: dict[str, list[Path]] = {}
    for split, folders in SPLITS.items():
        print(f"  {split}")
        for folder in folders:
            suffixes = {".png"} if folder == "masks" else IMAGE_SUFFIXES
            paths = iter_files(data_root / split / folder, suffixes)
            print(f"    {folder}: {len(paths)} files")
            if folder == "images":
                split_images[split] = paths
                sizes = summarize_sizes(paths, size_sample)
                size_text = ", ".join(f"{size} x {count}" for size, count in sizes.most_common(5))
                print(f"    sampled sizes: {size_text or 'none'}")
    return split_images


def check_segmentation_pairs(data_root: Path, limit: int) -> None:
    print("\nImage/mask size checks")
    for split in ("train_seg", "val"):
        image_dir = data_root / split / "images"
        mask_dir = data_root / split / "masks"
        images = iter_files(image_dir, {".jpg", ".jpeg"})
        mismatches = []
        missing_masks = []
        for image_path in images[:limit]:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                missing_masks.append(image_path.name)
                continue
            if image_size(image_path) != image_size(mask_path):
                mismatches.append((image_path.name, image_size(image_path), image_size(mask_path)))
        print(f"  {split}: checked={min(len(images), limit)}, missing_masks={len(missing_masks)}, size_mismatches={len(mismatches)}")
        if missing_masks:
            print(f"    first missing masks: {missing_masks[:5]}")
        if mismatches:
            print(f"    first mismatches: {mismatches[:5]}")


def inspect_masks(data_root: Path, sample_masks: int, frequency_masks: int, random_seed: int) -> None:
    print("\nMask id checks")
    mask_paths = iter_files(data_root / "train_seg" / "masks", {".png"})
    val_mask_paths = iter_files(data_root / "val" / "masks", {".png"})
    checked_paths = mask_paths[:sample_masks] + val_mask_paths[:sample_masks]
    for path in checked_paths:
        mask = decode_rgb_mask(path)
        ids = np.unique(mask)
        is_valid, invalid = validate_mask_ids(mask, allow_ignore=True)
        warning = f" WARNING invalid={invalid.tolist()}" if not is_valid else ""
        print(f"  {path.relative_to(data_root)} unique_ids={ids[:20].tolist()} count={len(ids)}{warning}")

    sample_size = min(frequency_masks, len(mask_paths))
    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.choice(len(mask_paths), size=sample_size, replace=False) if sample_size else []
    sampled_paths = [mask_paths[int(index)] for index in sampled_indices]

    print(
        "\nApproximate segmentation-id frequency from "
        f"{sample_size} uniformly sampled train_seg masks "
        f"(seed={random_seed})"
    )
    counts = np.zeros(IGNORE_ID + 1, dtype=np.int64)
    foreground_percentages: list[float] = []
    multi_foreground_masks = 0
    foreground_id_histogram: Counter[int] = Counter()

    for path in sampled_paths:
        mask = decode_rgb_mask(path)
        ids, freq = np.unique(mask, return_counts=True)
        foreground_ids = [int(mask_id) for mask_id in ids if 1 <= int(mask_id) <= NUM_CLASSES]
        foreground_pixels = int(np.isin(mask, foreground_ids).sum()) if foreground_ids else 0
        foreground_percentages.append(100.0 * foreground_pixels / mask.size)
        if len(foreground_ids) > 1:
            multi_foreground_masks += 1
        foreground_id_histogram.update(foreground_ids)

        for mask_id, pixel_count in zip(ids, freq):
            if 0 <= int(mask_id) <= IGNORE_ID:
                counts[int(mask_id)] += int(pixel_count)
    present = np.flatnonzero(counts)
    foreground = [int(mask_id) for mask_id in present if 1 <= int(mask_id) <= NUM_CLASSES]
    percentages = np.array(foreground_percentages, dtype=np.float64)
    print(f"  unique foreground ids observed: {len(foreground)}")
    print(f"  foreground id histogram by mask presence: {dict(sorted(foreground_id_histogram.items()))}")
    print(f"  masks containing more than one foreground class: {multi_foreground_masks}")
    if len(percentages):
        print(f"  average foreground pixel percentage: {float(percentages.mean()):.2f}%")
        print(f"  min foreground pixel percentage: {float(percentages.min()):.2f}%")
        print(f"  max foreground pixel percentage: {float(percentages.max()):.2f}%")
    else:
        print("  average foreground pixel percentage: n/a")
        print("  min foreground pixel percentage: n/a")
        print("  max foreground pixel percentage: n/a")
    print(f"  background pixels: {int(counts[0])}")
    print(f"  ignore pixels: {int(counts[IGNORE_ID])}")
    top = sorted(((int(mask_id), int(counts[mask_id])) for mask_id in foreground), key=lambda item: item[1], reverse=True)[:10]
    print(f"  top foreground ids: {top}")


def audit_all_train_seg_masks(data_root: Path) -> None:
    """Audit every train_seg example for foreground ids and metadata consistency."""
    rows = load_json(data_root / "metadata" / "train_seg.json")
    if not isinstance(rows, list):
        raise TypeError("metadata/train_seg.json should contain a list of rows")

    masks_with_multiple_foreground_ids = 0
    masks_with_no_foreground_ids = 0
    metadata_mismatch_count = 0
    invalid_id_example_count = 0
    metadata_mismatches: list[dict[str, object]] = []
    invalid_id_examples: list[dict[str, object]] = []
    foreground_id_by_mask_presence: Counter[int] = Counter()
    foreground_pixel_counts: Counter[int] = Counter()

    for index, row in enumerate(rows, start=1):
        mask_path = data_root / str(row["mask"])
        expected_segmentation_id = int(row["class_id"]) + 1
        metadata_segmentation_id = int(row.get("segmentation_id", expected_segmentation_id))
        mask = decode_rgb_mask(mask_path)
        ids, counts = np.unique(mask, return_counts=True)
        ids_int = [int(mask_id) for mask_id in ids]
        foreground_ids = [mask_id for mask_id in ids_int if 1 <= mask_id <= NUM_CLASSES]

        if len(foreground_ids) > 1:
            masks_with_multiple_foreground_ids += 1
        if not foreground_ids:
            masks_with_no_foreground_ids += 1

        foreground_id_by_mask_presence.update(foreground_ids)
        for mask_id, pixel_count in zip(ids_int, counts):
            if 1 <= mask_id <= NUM_CLASSES:
                foreground_pixel_counts[mask_id] += int(pixel_count)

        invalid_ids = [mask_id for mask_id in ids_int if mask_id not in range(NUM_CLASSES + 1) and mask_id != IGNORE_ID]
        if invalid_ids:
            invalid_id_example_count += 1
            if len(invalid_id_examples) < 10:
                invalid_id_examples.append({"mask": str(row["mask"]), "invalid_ids": invalid_ids})

        if metadata_segmentation_id != expected_segmentation_id or foreground_ids != [expected_segmentation_id]:
            metadata_mismatch_count += 1
            if len(metadata_mismatches) < 20:
                metadata_mismatches.append(
                    {
                        "mask": str(row["mask"]),
                        "class_id": int(row["class_id"]),
                        "metadata_segmentation_id": metadata_segmentation_id,
                        "expected_segmentation_id": expected_segmentation_id,
                        "foreground_ids_in_mask": foreground_ids,
                    }
                )

        if index % 500 == 0:
            print(f"  audited {index}/{len(rows)} masks")

    print("\nFull train_seg audit")
    print(f"  total masks audited: {len(rows)}")
    print(f"  masks containing >1 foreground segmentation_id: {masks_with_multiple_foreground_ids}")
    print(f"  masks containing no foreground segmentation_id: {masks_with_no_foreground_ids}")
    print(f"  unique foreground ids observed: {len(foreground_id_by_mask_presence)}")
    print(f"  metadata/id consistency mismatches: {metadata_mismatch_count}")
    if metadata_mismatches:
        print(f"    first mismatches: {metadata_mismatches[:5]}")
    print(f"  masks with invalid ids: {invalid_id_example_count}")
    if invalid_id_examples:
        print(f"    first invalid id examples: {invalid_id_examples[:5]}")

    by_presence = dict(sorted(foreground_id_by_mask_presence.items()))
    by_pixels_top = foreground_pixel_counts.most_common(20)
    print(f"  distribution of foreground ids by mask presence: {by_presence}")
    print(f"  top foreground ids by pixel count: {by_pixels_top}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--size-sample", type=int, default=200)
    parser.add_argument("--pair-check-limit", type=int, default=1000)
    parser.add_argument("--sample-masks", type=int, default=5)
    parser.add_argument("--frequency-masks", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=164)
    parser.add_argument("--audit-all-train-seg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    print(f"Dataset root: {data_root}")
    print_metadata_summary(data_root)
    print_split_counts(data_root, args.size_sample)
    check_segmentation_pairs(data_root, args.pair_check_limit)
    inspect_masks(data_root, args.sample_masks, args.frequency_masks, args.random_seed)
    if args.audit_all_train_seg:
        audit_all_train_seg_masks(data_root)


if __name__ == "__main__":
    main()
