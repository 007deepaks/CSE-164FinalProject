# CSE 164 Final Project - Deepak Sah

Semi-supervised image classification plus semantic segmentation for the CSE 164 Kaggle competition.

## Final Result

Best Kaggle submission:

```text
resnext116_cls448_blend032_thr05625_multicrop
Kaggle score: 0.22510
```

The final system used:

- A from-scratch ResNeXt-50 32x4d multi-task model for binary foreground segmentation.
- A from-scratch ConvNeXt-style EMA classifier for auxiliary class probabilities.
- Probability-space class blending:

```text
final_class_probs = 0.68 * ResNeXt_probs + 0.32 * ConvNeXt_probs
```

- Foreground threshold: `0.5625`
- Test-time augmentation: `multi_crop`

No pretrained model weights, pretrained backbones, public checkpoints, foundation-model features, or external training data were used. All final models were trained from scratch using only the provided competition data.

## Data

The Kaggle data is expected under:

```text
data/raw/
|-- metadata/
|-- test/
|-- train_labeled/
|-- train_seg/
|-- train_unlabeled/
`-- val/
```

The final supervised models used:

- `train_seg` for images with segmentation masks and class labels.
- `train_labeled` for images with class labels only.
- `val` for model selection, threshold tuning, and local scoring.
- `test` only for final submission generation.

The `train_unlabeled` split was used in pseudo-labeling and FixMatch experiments, but those experiments did not produce the final best submission.

## Final Model

### ResNeXt Multi-Task Model

The main model is a from-scratch ResNeXt-50 32x4d shared encoder with:

- A U-Net-style binary foreground segmentation decoder.
- A 300-way image classifier.
- Mask-guided classification pooling, where foreground probabilities weight deep encoder features before classification.

The model predicts a binary foreground mask. At inference time, foreground pixels are assigned `predicted_class_id + 1`, and background pixels are assigned `0`.

Best supervised ResNeXt validation checkpoint:

```text
val_auto=0.2146
mIoU=0.1863
bin_iou=0.6111
oracle_mIoU=0.5887
rare_mIoU=0.1137
macro_acc=0.3661
Kaggle=0.21444
```

### ConvNeXt Auxiliary Classifier

The auxiliary classifier is a from-scratch ConvNeXt-style classifier trained only for 300-way class prediction. It used:

- Image size 448 for the best auxiliary classifier.
- Strong classification augmentation.
- Mask-guided crops from segmentation examples.
- EMA checkpointing.

This classifier was weaker than the ResNeXt classifier alone on local validation, but it made different errors. Blending it with the ResNeXt class probabilities improved Kaggle score.

## Training Commands

### Train Final ResNeXt Multi-Task Model

```bash
python -m src.training.train_multitask \
  --data-root "$DATA_ROOT" \
  --architecture resnext50_32x4d \
  --stage joint \
  --epochs 120 \
  --warmup-epochs 3 \
  --image-size 384 \
  --num-segmentation-classes 1 \
  --seg-batch-size 12 \
  --cls-batch-size 64 \
  --val-batch-size 32 \
  --num-workers 12 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-6 \
  --weight-decay 5e-2 \
  --label-smoothing 0.1 \
  --segmentation-loss-weight 1.0 \
  --dice-loss-weight 1.0 \
  --seg-classification-loss-weight 1.0 \
  --cls-loss-weight 1.0 \
  --gradient-clip 2.0 \
  --ema-decay 0.9998 \
  --weighted-combined-sampling \
  --class-only-sample-weight 1.0 \
  --mask-sample-weight 2.5 \
  --validation-threshold 0.50 \
  --checkpoint-dir "$DRIVE_OUTPUTS/checkpoints/resnext50_32x4d_multitask_384_scratch_e130"
```

### Train Auxiliary ConvNeXt Classifier

```bash
python -u -m src.training.train_classification \
  --data-root "$DATA_ROOT" \
  --split train_combined \
  --image-size 448 \
  --epochs 100 \
  --batch-size 24 \
  --num-workers 8 \
  --learning-rate 2e-4 \
  --min-learning-rate 1e-6 \
  --weight-decay 5e-2 \
  --label-smoothing 0.05 \
  --mixup-alpha 0.1 \
  --cutmix-alpha 0.5 \
  --mix-prob 0.25 \
  --random-erasing-prob 0.10 \
  --ema-decay 0.9998 \
  --grad-clip-norm 1.0 \
  --augment-policy strong \
  --include-seg-crops \
  --crop-padding 0.20 \
  --balanced-class-batches \
  --base-channels 96 \
  --depths "3,3,9,3" \
  --mlp-ratio 4 \
  --drop-path 0.02 \
  --checkpoint-dir "$DRIVE_OUTPUTS/checkpoints/convnext_classifier_448_ema_finetune"
```

## Final Inference Command

```bash
BASE_CKPT="$DRIVE_OUTPUTS/checkpoints/resnext50_32x4d_multitask_384_scratch_e130/best_multitask.pt"
CLS_CKPT="$DRIVE_OUTPUTS/checkpoints/convnext_classifier_448_ema_finetune/best_ema_classification.pt"

python -m src.training.predict_multitask \
  --checkpoint "$BASE_CKPT" \
  --classifier-checkpoint "$CLS_CKPT" \
  --classifier-blend-weight 0.32 \
  --data-root "$DATA_ROOT" \
  --output "$DRIVE_OUTPUTS/submissions/resnext116_cls448_blend032_thr05625_multicrop.csv" \
  --image-size 384 \
  --batch-size 4 \
  --num-workers 8 \
  --seg-threshold 0.5625 \
  --tta multi_crop
```

## Useful Evaluation Commands

Evaluate the final blend on validation:

```bash
python -m src.training.evaluate_multitask \
  --checkpoint "$BASE_CKPT" \
  --classifier-checkpoint "$CLS_CKPT" \
  --classifier-blend-weight 0.32 \
  --data-root "$DATA_ROOT" \
  --image-size 384 \
  --batch-size 6 \
  --num-workers 8 \
  --seg-threshold 0.5625 \
  --tta multi_crop
```

Tune foreground threshold:

```bash
python -m src.training.tune_multitask_threshold \
  --seg-checkpoint "$BASE_CKPT" \
  --classifier-checkpoint "$CLS_CKPT" \
  --classifier-blend-weight 0.32 \
  --data-root "$DATA_ROOT" \
  --image-size 384 \
  --batch-size 6 \
  --num-workers 8 \
  --thresholds "0.50,0.525,0.55,0.5625,0.575,0.60,0.625,0.65" \
  --tta hflip
```

Visualize 20 validation predictions:

```bash
python -m src.visualization.visualize_val_predictions \
  --checkpoint "$BASE_CKPT" \
  --classifier-checkpoint "$CLS_CKPT" \
  --classifier-blend-weight 0.32 \
  --data-root "$DATA_ROOT" \
  --split val \
  --image-size 384 \
  --num-samples 20 \
  --output-dir "$DRIVE_OUTPUTS/figures/final_blend_val20" \
  --seg-threshold 0.5625 \
  --tta hflip
```

## Methods Tried

Major experiments included:

- Small U-Net segmentation baselines.
- ConvNeXt-style classifier baselines.
- ConvNeXt tiny multi-task segmentation/classification models.
- ResNet-50 multi-task models.
- ResNeXt-50 32x4d multi-task models.
- Classifier warmup followed by joint multi-task training.
- Pure joint multi-task training.
- Weighted combined sampling between class-only and mask examples.
- BCE + Dice segmentation loss.
- EMA checkpointing.
- Segmentation threshold tuning.
- Offline high-confidence pseudo-labeling.
- Online FixMatch-style SSL.
- Distribution alignment for pseudo-label probabilities.
- Classifier-only fine-tuning.
- Foreground CutMix and classifier-only MixUp/CutMix.
- Dedicated ConvNeXt classifier training.
- Mask-guided classifier crops.
- Probability-space ensembling.
- Horizontal flip TTA.
- Multi-crop TTA.

The most important finding was that the ResNeXt model learned strong foreground shapes, while many remaining errors came from selecting the wrong class. The final improvement came from keeping the ResNeXt segmentation model intact and blending its class probabilities with a separately trained ConvNeXt classifier.

## Experiment Highlights

| Model / Method | Validation Automated Score | Kaggle Score |
| --- | ---: | ---: |
| ConvNeXt tiny multi-task v2 | 0.1468 | 0.14347 |
| ResNet-50 multi-task | 0.1812 | 0.17163 |
| ResNet-50 offline pseudo-label fine-tune | 0.1846 | not final |
| ResNeXt-50 32x4d multi-task | 0.2146 | 0.21444 |
| ResNeXt + ConvNeXt blend, hflip | not final | 0.22355 |
| ResNeXt + ConvNeXt blend, multi-crop | best | 0.22510 |

## Software

The project used:

- Python
- PyTorch
- NumPy
- pandas
- PIL
- Kaggle starter scoring and submission utilities
- Google Colab for GPU training and inference

The `starter/` folder contains the provided Kaggle utilities for RLE encoding, validation, and local scoring.

## Repository Layout

```text
src/
|-- data/             dataset classes and augmentation
|-- metrics/          classification and segmentation metrics
|-- models/           ConvNeXt, ResNet, ResNeXt, and multi-task models
|-- submission/       submission sanity checks
|-- training/         training, evaluation, prediction, pseudo-labeling
|-- utils/            mask decoding, validation, and RLE helpers
`-- visualization/   mask and prediction visualization
```

