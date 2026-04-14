# src/utils/CLAUDE.md

## data_loaders.py

Handles all image I/O for both pipelines.

**Key functions:**

| Function | Purpose |
|---|---|
| `load_paths_labels(file_path)` | Reads a tab-separated `.txt` file → `(paths_array, labels_array)`. Skips missing files with a warning. |
| `load_image(path)` | Loads a single BGR image with `cv2`. Returns `None` on failure. |
| `resize_image(image, target_size)` | Resizes to `(height, width)`; defaults to `cfg.img_size[:2]`. |
| `apply_model_preprocessing(image, model_name)` | Applies backbone-specific preprocessing (VGG19, Inception, ResNet, Xception, EfficientNet). |
| `MemoryEfficientDataGenerator` | Iterator for the CNN pipeline. Loads images in batches on demand — never materialises the full dataset in RAM. Supports optional augmentation and Mixup. |

**Mixup:** When `cfg.use_mixup=True`, each training batch is blended with a randomly permuted copy of itself using a Beta(0.4, 0.4) coefficient. Applied only when `augment_fn` is set (training mode).

The text file format (used by `res/train_files.txt`, `val_files.txt`, `test_files.txt`):
```
/absolute/path/to/image.jpg\t<integer_label>
```
Labels are integers 0–6 corresponding to HAM10000 classes.

## fold_utils.py

`save_fold_results(fold_results, result_dir, classifier_name)` — takes a list of
per-fold metric dicts and writes:
- `fold_results_summary.csv` — one row per fold (accuracy, precision, recall, F1).
- `fold_results_detailed.json` — full per-fold dicts.
- `iteration_summary_stats.csv` — mean/std/min/max per iteration.

Called by both `cnn_classifier.py` and `feature_extraction.py` at the end of each
cross-validation loop.

## metadata_extractor.py

`MetadataFeatureExtractor` encodes clinical metadata from `res/metadata.csv` into a
fixed-length feature vector that can be concatenated with CNN features when
`USE_METADATA=True`.

**Encoded fields:**
- `age` — StandardScaler normalized + 5 age-bin dummies + squared term.
- `sex` — one-hot encoded.
- `localization` — one-hot encoded.
- Engineered: `high_risk_age`, `sun_exposed_area`, `extremities`, `trunk_area`.

`dx_type` is intentionally **excluded** — it encodes how the ground-truth label was
obtained (biopsy, consensus, follow-up) which is unavailable at clinical prediction
time. Including it would be target leakage.

**Fit on train+val only.** Never call `.fit()` on the full dataset — the fitted
StandardScaler would see test-set statistics.

**Usage:**
```python
from utils.metadata_extractor import extract_metadata_for_paths, combine_cnn_and_metadata_features
meta_features = extract_metadata_for_paths(image_paths, metadata_csv)
X_combined = combine_cnn_and_metadata_features(cnn_features, meta_features)
```

The fitted extractor is saved/loaded with `joblib` alongside the classifier.

## calibration.py

`expected_calibration_error(y_true, y_prob, n_bins=15)` — computes ECE (Guo et al.,
2017) by binning samples by confidence and measuring the gap between mean confidence
and accuracy in each bin. Returns a scalar in [0, 1]; lower is better.

Used by both `cnn_classifier.py` and `feature_extraction.py` after evaluation.
