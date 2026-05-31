# Starter Utilities

These scripts are intentionally small. They validate the Kaggle CSV/RLE
submission format and let students score on the public validation split.

Decoded prediction masks may use ids `0..300` only. In the CSV RLE column, use `0` for an all-background predicted mask. The id `1000` is reserved for ground-truth ignore regions.

Create a Kaggle CSV baseline submission:

```bash
python starter/make_sample_submission_csv.py \
  --data-root /path/to/release \
  --split test \
  --output sample_submission.csv
```

Validate and score on the validation split:

```bash
python starter/make_sample_submission_csv.py \
  --data-root /path/to/release \
  --split val \
  --output val_sample_submission.csv

python starter/validate_submission_csv.py \
  --submission val_sample_submission.csv \
  --data-root /path/to/release \
  --split val
```

Validate a hidden-test CSV before uploading to Kaggle:

```bash
python starter/validate_submission_csv.py \
  --submission sample_submission.csv \
  --data-root /path/to/release \
  --split test
```
