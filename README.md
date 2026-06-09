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

## Shared ConvNeXt Multi-Task Pipeline

This is the recommended pipeline before pseudo-labeling. It trains one
from-scratch ConvNeXt encoder shared by:

- a `300`-class image classification head,
- a binary foreground/background segmentation head.

The final semantic segmentation mask maps foreground pixels to
`predicted_class_id + 1`. This matches the observed labeled segmentation data,
where each training mask contains one foreground class and
`segmentation_id = class_id + 1`.

No pretrained weights, pretrained backbones, public checkpoints, or foundation
model features are used.

Quick smoke training run:

```powershell
python -m src.training.train_multitask --data-root data/raw --epochs 1 --image-size 320 --model-size small --seg-batch-size 1 --cls-batch-size 2 --max-seg-samples 4 --max-cls-samples 8 --max-val-samples 8
```

Tiny overfit debug run:

```powershell
python -m src.training.train_multitask --data-root data/raw --debug-overfit --epochs 30 --image-size 320 --model-size tiny --seg-batch-size 2 --cls-batch-size 8 --learning-rate 1e-3
```

This disables augmentation, label smoothing, weight decay, and drop path, then
trains on 8 segmentation images and 32 classification images. Each epoch prints
training binary foreground IoU, binary pixel accuracy, oracle semantic mIoU,
segmentation-set class accuracy, and classification-set accuracy. If these do
not climb on the tiny subset, inspect labels, losses, output mapping, and the
training loop before launching another full VM run.

Recommended 1x V100 run:

```powershell
python -m src.training.train_multitask --data-root data/raw --epochs 40 --image-size 320 --model-size small --seg-batch-size 2 --cls-batch-size 16 --num-workers 4
```

If early full runs stay near random, try this lower-risk diagnostic training
run before scaling back up:

```powershell
python -m src.training.train_multitask --data-root data/raw --epochs 20 --image-size 320 --model-size tiny --seg-batch-size 4 --cls-batch-size 32 --num-workers 4 --learning-rate 1e-3 --weight-decay 1e-4 --drop-path 0.0 --no-random-crop --seg-classification-loss-weight 1.0 --cls-loss-weight 1.0
```

Watch `bin_iou` and `oracle_mIoU` separately from `mIoU`. If they rise while
`mIoU` stays low, segmentation shape is learning and classification is the
bottleneck. If all three stay flat, inspect masks, foreground percentage, and
loss behavior before another long run.

One-command classifier warmup, then mixed joint fine-tuning:

```powershell
python -m src.training.train_multitask --data-root data/raw --stage warmup_joint --warmup-epochs 8 --joint-epochs 24 --image-size 320 --model-size tiny --seg-batch-size 4 --cls-batch-size 28 --num-workers 4 --learning-rate 2e-4 --warmup-learning-rate 3e-4 --weight-decay 1e-4 --drop-path 0.0 --no-random-crop --balanced-class-batches --seg-classification-loss-weight 1.0 --cls-loss-weight 1.0 --validate-every 2 --full-val-every 4 --quick-val-samples 150 --checkpoint-dir outputs/checkpoints/warmup_joint_tiny
```

The warmup stage trains classification on a combined 10,500-image labeled set:
`train_labeled` plus `train_seg` class labels. The joint stage resumes the best
warmup weights and uses mixed optimizer steps containing one segmentation batch
and one classification batch. Full validation runs every `--full-val-every`
epochs; quick validation uses `--quick-val-samples` on intermediate validation
epochs. Non-finite losses or gradients are reported and skipped.

Training uses synchronized random resized crop and horizontal flip for
segmentation images/masks, image-only color jitter/blur, AMP on CUDA, AdamW,
cosine LR decay, weighted segmentation CE, Dice loss, and classification CE
with label smoothing.

Outputs:

```text
outputs/checkpoints/best_multitask.pt
outputs/checkpoints/latest_multitask.pt
outputs/checkpoints/multitask_history.json
```

The best checkpoint is selected by the validation automated score:

```text
0.70 * segmentation_score + 0.20 * classification_macro_accuracy
```

The logged validation metrics include automated score, segmentation score,
mean IoU, boundary F-score, rare-class mIoU, accuracy, and macro accuracy.

Evaluate the best multi-task checkpoint:

```powershell
python -m src.training.evaluate_multitask --checkpoint outputs/checkpoints/best_multitask.pt --data-root data/raw --image-size 320
```

Visualize predictions against labeled validation masks:

```powershell
python -m src.visualization.visualize_val_predictions --checkpoint outputs/checkpoints/best_multitask.pt --data-root data/raw --split val --num-samples 12
```

Validation panels contain image, ground truth, prediction, prediction overlay,
and a difference view. The command can also run on `--split test`, but test
panels only show raw predictions and overlays because hidden test labels are
not available and must not be inferred.

Generate a Kaggle test submission:

```powershell
python -m src.training.predict_multitask --checkpoint outputs/checkpoints/best_multitask.pt --data-root data/raw --image-size 320 --output outputs/submissions/submission.csv
```

Generate a TTA submission from the one-command pipeline:

```powershell
python -m src.training.predict_multitask --checkpoint outputs/checkpoints/warmup_joint_tiny/joint/best_multitask.pt --data-root data/raw --image-size 320 --batch-size 4 --num-workers 4 --tta hflip --output outputs/submissions/warmup_joint_tiny_tta.csv
```

For a quick partial inference smoke check:

```powershell
python -m src.training.predict_multitask --checkpoint outputs/checkpoints/best_multitask.pt --data-root data/raw --image-size 320 --max-test-samples 8 --no-validate
```

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

## First Classification Baseline

This baseline trains a small ConvNeXt-style classifier from scratch for the 300
image-level classes. It uses a patch stem, depthwise convolution blocks,
LayerNorm, GELU, residual connections, global average pooling, and a linear
classifier. No pretrained weights are used.

Quick smoke training run:

```powershell
python -m src.training.train_classification --data-root data/raw --epochs 1 --batch-size 2 --base-channels 16 --depths 1,1,1,1 --max-train-samples 8 --max-val-samples 8
```

Normal first classifier run:

```powershell
python -m src.training.train_classification --data-root data/raw --epochs 20 --batch-size 32 --base-channels 48 --depths 2,2,4,2
```

Evaluate the best classifier checkpoint:

```powershell
python -m src.training.evaluate_classification --checkpoint outputs/checkpoints/best_classification.pt --data-root data/raw
```

Predict test image classes:

```powershell
python -m src.training.predict_classification --checkpoint outputs/checkpoints/best_classification.pt --data-root data/raw --output outputs/predictions/test_class_predictions.csv
```

Generate a segmentation submission using a binary segmentation checkpoint and
classifier class predictions:

```powershell
python -m src.training.predict_test --checkpoint outputs/checkpoints/best_segmentation.pt --data-root data/raw --class-csv outputs/predictions/test_class_predictions.csv --output outputs/submissions/submission.csv
```

## Dedicated Classifier and Threshold Tuning

The current multi-task segmentation model can learn foreground shape, but final
semantic ids depend heavily on the classifier. This path trains a stronger
classifier separately, then uses it for final submission class ids.

Train on all supervised class labels, including mask-guided crops from
`train_seg`:

```powershell
python -m src.training.train_classification --data-root data/raw --split train_combined --include-seg-crops --image-size 320 --epochs 60 --batch-size 32 --num-workers 4 --base-channels 96 --depths 3,3,9,3 --learning-rate 3e-4 --weight-decay 1e-4 --label-smoothing 0.0 --drop-path 0.0 --no-random-crop --balanced-class-batches --checkpoint-dir outputs/checkpoints/classifier_combined_crops
```

Tune the segmentation foreground threshold on validation:

```powershell
python -m src.training.tune_multitask_threshold --seg-checkpoint outputs/checkpoints/joint_from_warmup/best_multitask.pt --classifier-checkpoint outputs/checkpoints/classifier_combined_crops/best_classification.pt --data-root data/raw --image-size 320 --batch-size 4 --num-workers 4 --tta hflip
```

Generate a submission using the segmentation checkpoint for foreground masks
and the dedicated classifier for class ids:

```powershell
python -m src.training.predict_multitask --checkpoint outputs/checkpoints/joint_from_warmup/best_multitask.pt --classifier-checkpoint outputs/checkpoints/classifier_combined_crops/best_classification.pt --data-root data/raw --image-size 320 --batch-size 4 --num-workers 4 --tta hflip --seg-threshold BEST_THRESHOLD --output outputs/submissions/seg_joint_classifier_crops_tta.csv
```

Supervised classifier v2 fine-tuning before returning to pseudo-labeling:

```powershell
python -m src.training.train_classification --data-root data/raw --resume-checkpoint outputs/checkpoints/classifier_combined_crops/best_classification.pt --split train_combined --include-seg-crops --image-size 320 --epochs 25 --batch-size 24 --num-workers 2 --learning-rate 5e-5 --min-learning-rate 1e-6 --weight-decay 5e-2 --label-smoothing 0.10 --mixup-alpha 0.20 --cutmix-alpha 1.00 --mix-prob 0.50 --random-erasing-prob 0.25 --ema-decay 0.999 --grad-clip-norm 1.0 --no-random-crop --checkpoint-dir outputs/checkpoints/classifier_supervised_v2
```

Evaluate the EMA classifier with hflip or multi-crop TTA:

```powershell
python -m src.training.evaluate_classification --checkpoint outputs/checkpoints/classifier_supervised_v2/best_ema_classification.pt --data-root data/raw --image-size 320 --batch-size 32 --num-workers 2 --tta hflip
python -m src.training.evaluate_classification --checkpoint outputs/checkpoints/classifier_supervised_v2/best_ema_classification.pt --data-root data/raw --image-size 320 --batch-size 16 --num-workers 2 --tta multi_crop
```

Classifier checkpoints can be ensembled by listing multiple paths after one
`--checkpoint` or `--classifier-checkpoint` flag:

```powershell
python -m src.training.evaluate_classification --checkpoint outputs/checkpoints/classifier_combined_crops/best_classification.pt outputs/checkpoints/classifier_supervised_v2/best_ema_classification.pt --data-root data/raw --image-size 320 --batch-size 16 --num-workers 2 --tta hflip
```

Retune the segmentation threshold and generate a submission with the improved
classifier:

```powershell
python -m src.training.tune_multitask_threshold --seg-checkpoint outputs/checkpoints/joint_from_warmup/best_multitask.pt --classifier-checkpoint outputs/checkpoints/classifier_supervised_v2/best_ema_classification.pt --data-root data/raw --image-size 320 --batch-size 4 --num-workers 4 --tta hflip --thresholds 0.60,0.65,0.70,0.75,0.80,0.85,0.90
python -m src.training.predict_multitask --checkpoint outputs/checkpoints/joint_from_warmup/best_multitask.pt --classifier-checkpoint outputs/checkpoints/classifier_supervised_v2/best_ema_classification.pt --data-root data/raw --image-size 320 --batch-size 4 --num-workers 4 --tta hflip --seg-threshold BEST_THRESHOLD --output outputs/submissions/supervised_v2_tta.csv
```

## Classifier FixMatch on Unlabeled Images

If the segmentation oracle mIoU is much higher than semantic mIoU, class
prediction is the main bottleneck. Fine-tune the dedicated classifier with
FixMatch using `train_unlabeled` while keeping supervised classification loss
on `train_labeled + train_seg` labels and mask-guided crops.

Start with a high confidence threshold:

```powershell
python -m src.training.train_fixmatch_classification --data-root data/raw --resume-checkpoint outputs/checkpoints/classifier_combined_crops/best_classification.pt --image-size 320 --epochs 20 --batch-size 24 --unlabeled-batch-size 32 --weak-forward-batch-size 16 --strong-forward-batch-size 16 --num-workers 0 --learning-rate 1e-4 --weight-decay 1e-4 --confidence-threshold 0.95 --unlabeled-loss-weight 1.0 --split train_combined --include-seg-crops --no-random-crop --checkpoint-dir outputs/checkpoints/classifier_fixmatch_095
```

The log reports `accept`, the fraction of unlabeled images whose pseudo-label
confidence was high enough to train on. If `accept` stays near zero for several
epochs, lower the threshold for a second round:

```powershell
python -m src.training.train_fixmatch_classification --data-root data/raw --resume-checkpoint outputs/checkpoints/classifier_fixmatch_095/best_fixmatch_classification.pt --image-size 320 --epochs 15 --batch-size 24 --unlabeled-batch-size 32 --weak-forward-batch-size 16 --strong-forward-batch-size 16 --num-workers 0 --learning-rate 5e-5 --weight-decay 1e-4 --confidence-threshold 0.90 --unlabeled-loss-weight 0.75 --split train_combined --include-seg-crops --no-random-crop --checkpoint-dir outputs/checkpoints/classifier_fixmatch_090
```

After FixMatch, reuse the same segmentation checkpoint and retune the foreground
threshold with the new classifier:

```powershell
python -m src.training.tune_multitask_threshold --seg-checkpoint outputs/checkpoints/joint_from_warmup/best_multitask.pt --classifier-checkpoint outputs/checkpoints/classifier_fixmatch_095/best_fixmatch_classification.pt --data-root data/raw --image-size 320 --batch-size 4 --num-workers 4 --tta hflip --thresholds 0.60,0.65,0.70,0.75,0.80,0.85,0.90
```

Generate a TTA submission with the best tuned threshold:

```powershell
python -m src.training.predict_multitask --checkpoint outputs/checkpoints/joint_from_warmup/best_multitask.pt --classifier-checkpoint outputs/checkpoints/classifier_fixmatch_095/best_fixmatch_classification.pt --data-root data/raw --image-size 320 --batch-size 4 --num-workers 4 --tta hflip --seg-threshold BEST_THRESHOLD --output outputs/submissions/seg_joint_fixmatch_tta.csv
```
