"""Mask decoding, validation, and Kaggle RLE helpers.

Competition facts:
- Ground-truth PNG masks encode ids as segmentation_id = R + G * 256.
- Ground-truth masks may contain 1000 for ignore regions.
- Predicted masks may contain only 0..300.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

NUM_CLASSES = 300
BACKGROUND_ID = 0
IGNORE_ID = 1000
VALID_PREDICTION_IDS = set(range(NUM_CLASSES + 1))
VALID_GROUND_TRUTH_IDS = VALID_PREDICTION_IDS | {IGNORE_ID}


def decode_rgb_mask(mask_path: str | Path) -> np.ndarray:
    """Decode an RGB PNG mask into a 2D integer segmentation-id array."""
    with Image.open(mask_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint16)
    return rgb[:, :, 0] + rgb[:, :, 1] * 256


def validate_mask_ids(
    mask_ids: np.ndarray,
    *,
    allow_ignore: bool = False,
    num_classes: int = NUM_CLASSES,
) -> tuple[bool, np.ndarray]:
    """Return whether ids are valid and the sorted invalid ids."""
    ids = np.unique(mask_ids.astype(np.int64, copy=False))
    valid = set(range(num_classes + 1))
    if allow_ignore:
        valid.add(IGNORE_ID)
    invalid = np.array([mask_id for mask_id in ids if int(mask_id) not in valid], dtype=np.int64)
    return len(invalid) == 0, invalid


def validate_prediction_mask(mask_ids: np.ndarray, *, num_classes: int = NUM_CLASSES) -> None:
    """Raise ValueError if a predicted mask contains ids outside 0..num_classes."""
    is_valid, invalid = validate_mask_ids(mask_ids, allow_ignore=False, num_classes=num_classes)
    if not is_valid:
        raise ValueError(f"Prediction mask has invalid ids: {invalid.tolist()}")


def encode_mask_to_rle(mask_ids: np.ndarray, *, all_background_value: str = "0") -> str:
    """Encode a predicted id mask as Kaggle row-major 1-indexed RLE triples."""
    validate_prediction_mask(mask_ids)
    flat = np.asarray(mask_ids, dtype=np.int64).reshape(-1)
    nonzero_indices = np.flatnonzero(flat != BACKGROUND_ID)
    if len(nonzero_indices) == 0:
        return all_background_value

    values = flat[nonzero_indices]
    run_starts = np.ones(len(nonzero_indices), dtype=bool)
    run_starts[1:] = (
        (nonzero_indices[1:] != nonzero_indices[:-1] + 1)
        | (values[1:] != values[:-1])
    )
    start_positions = np.flatnonzero(run_starts)
    end_positions = np.r_[start_positions[1:], len(nonzero_indices)]

    parts: list[str] = []
    for start_pos, end_pos in zip(start_positions, end_positions):
        start = int(nonzero_indices[start_pos]) + 1
        length = int(end_pos - start_pos)
        value = int(values[start_pos])
        parts.extend([str(start), str(length), str(value)])
    return " ".join(parts)


def decode_rle_to_mask(rle: object, height: int, width: int) -> np.ndarray:
    """Decode Kaggle RLE triples into a 2D prediction mask."""
    total = int(height) * int(width)
    mask = np.zeros(total, dtype=np.uint16)
    if rle is None or str(rle).strip() in {"", "0", "nan"}:
        return mask.reshape((int(height), int(width)))

    tokens = [int(token) for token in str(rle).split()]
    if len(tokens) % 3 != 0:
        raise ValueError("RLE must contain start length value triples")

    used = np.zeros(total, dtype=bool)
    for start, length, value in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if start < 1 or length < 1:
            raise ValueError("RLE starts and lengths must be positive")
        if value < 1 or value > NUM_CLASSES:
            raise ValueError(f"RLE values must be in 1..{NUM_CLASSES}")
        begin = start - 1
        end = begin + length
        if end > total:
            raise ValueError("RLE run extends past the image size")
        if used[begin:end].any():
            raise ValueError("RLE runs must not overlap")
        used[begin:end] = True
        mask[begin:end] = value
    return mask.reshape((int(height), int(width)))
