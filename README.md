# CSE-164FinalProject

Repository foundation for the CSE 164 Final Project Kaggle competition:
semi-supervised image classification plus semantic segmentation.

The first priority is understanding the dataset and proving that the Kaggle
submission pipeline works. Do not use pretrained weights, pretrained
backbones, foundation model outputs, or public checkpoints for final models.

## Data Layout

Keep the Kaggle data untouched under:

```text
data/raw/
|-- metadata/
|-- test/
|-- train_labeled/
|-- train_seg/
|-- train_unlabeled/
`-- val/
```

`data/raw/` is ignored by git.

## Starter Code

The `starter/` folder comes from Kaggle and is kept unchanged.

- `starter/make_sample_submission_csv.py`
  - Creates an all-background baseline CSV for `val` or `test`.
  - Uses `class_id = 0` for every image and `segmentation_rle = 0`.
- `starter/validate_submission_csv.py`
  - Validates required columns, filenames, duplicates, class id range, image coverage, and RLE decodability.
  - Scores validation submissions because `val/` includes labels and masks.
  - Checks test submission format only because hidden test labels are private.
- `starter/kaggle_metric.py`
  - Implements the Kaggle-compatible RLE encoding/decoding and scoring logic.
  - Scores classification macro accuracy, mean IoU, boundary F-score, and rare-class mIoU.
- `starter/README.md`
  - Gives the official starter utility usage examples.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Inspect Data

Run a lightweight dataset inspection:

```powershell
python -m src.data.inspect_dataset --data-root data/raw
```

Useful options:

```powershell
python -m src.data.inspect_dataset --data-root data/raw --sample-masks 10 --frequency-masks 100
```

This checks metadata, split counts, sampled image sizes, image/mask size
matches, decoded mask ids, invalid ids, and approximate segmentation-id
frequency.

## Visualize Masks

Save image, decoded-mask, and overlay panels:

```powershell
python -m src.visualization.visualize_masks --data-root data/raw --split train_seg --num-samples 6
python -m src.visualization.visualize_masks --data-root data/raw --split val --num-samples 6
```

Outputs are saved under:

```text
outputs/figures/
```

## Submission Sanity Check

Generate and validate starter sample submissions:

```powershell
python -m src.submission.sanity_check_submission --data-root data/raw
```

This writes:

```text
outputs/submissions/val_sample_submission.csv
outputs/submissions/test_sample_submission.csv
```

Validate a submission manually:

```powershell
python starter/validate_submission_csv.py --submission outputs/submissions/test_sample_submission.csv --data-root data/raw --split test
```

Score a validation submission manually:

```powershell
python starter/validate_submission_csv.py --submission outputs/submissions/val_sample_submission.csv --data-root data/raw --split val
```

## Mask and RLE Utilities

Shared helpers live in `src/utils/masks.py`.

- Decode RGB masks with `segmentation_id = R + G * 256`.
- Validate ground-truth masks may contain `0`, `1..300`, and `1000`.
- Validate predicted masks only contain `0..300`.
- Encode predicted masks into Kaggle row-major, 1-indexed RLE triples.

Prediction masks must never use `1000`.

## First Segmentation Baseline

This baseline trains a compact U-Net-style CNN from scratch. It does not use
pretrained weights or pretrained backbones.

The default training target is now `--target-mode binary`. This matches the
observed `train_seg` structure: every segmentation mask contains background and
exactly one foreground `segmentation_id`, and that id is always
`class_id + 1`. During validation, binary foreground predictions are converted
back to semantic ids using the known validation `class_id`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Quick smoke training run on a tiny subset:

```powershell
python -m src.training.train_segmentation --data-root data/raw --epochs 1 --batch-size 1 --base-channels 8 --max-train-samples 2 --max-val-samples 2 --num-visualizations 1
```

Normal first baseline training run:

```powershell
python -m src.training.train_segmentation --data-root data/raw --epochs 10 --batch-size 8 --base-channels 32
```

Useful explicit binary-loss settings:

```powershell
python -m src.training.train_segmentation --data-root data/raw --epochs 10 --batch-size 8 --base-channels 32 --target-mode binary --background-weight 0.05 --foreground-weight 1.0
```

Outputs:

```text
outputs/checkpoints/best_segmentation.pt
outputs/checkpoints/latest_segmentation.pt
outputs/checkpoints/segmentation_history.json
outputs/figures/
```

Evaluate the best checkpoint on validation:

```powershell
python -m src.training.evaluate_segmentation --checkpoint outputs/checkpoints/best_segmentation.pt --data-root data/raw
```

Generate a model-based Kaggle test submission:

```powershell
python -m src.training.predict_test --checkpoint outputs/checkpoints/best_segmentation.pt --data-root data/raw --output outputs/submissions/submission.csv
```

The prediction script upsamples model logits back to each test image's original
resolution, converts the predicted ids to Kaggle RLE, and validates the full
test CSV with the starter validator.

For a binary checkpoint, test-time semantic ids require a class source. Until a
classifier is added, `predict_test.py` uses `--default-class-id 0`, which keeps
the CSV valid but is not a strong final test strategy. Later, pass a CSV with
`image,class_id` using `--class-csv`.
