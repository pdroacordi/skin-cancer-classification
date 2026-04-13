# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

All commands must be run from the `src/` directory, because modules use bare relative imports (`from config import ...`):

```bash
cd src

# Run both pipelines (default)
python main.py

# Run only CNN classifier pipeline
python main.py --pipeline cnn

# Run only feature extraction + classical ML pipeline
python main.py --pipeline feature-extraction

# Run with k-fold cross-validation
python main.py --cv

# Skip training (evaluation only)
python main.py --skip-train

# Create train/val/test splits from raw HAM10000 data
python main.py --create-splits --metadata path/to/HAM10000_metadata.csv \
  --images-dir1 path/to/part_1 --images-dir2 path/to/part_2

# Aggregate all experiment results into summary CSVs (res/all_models_general.csv, res/all_models_per_class.csv)
python -m analysis.aggregate_results
```

## Configuration

All experiment toggles live in `src/config.py`. Change them there before running — no CLI flags exist for most of them:

| Key | Options/Notes |
|---|---|
| `CNN_MODEL` | `'VGG19'`, `'Inception'`, `'ResNet'`, `'Xception'`, `'EfficientNet'` |
| `CLASSICAL_CLASSIFIER_MODEL` | `'RandomForest'`, `'XGBoost'`, `'AdaBoost'`, `'ExtraTrees'`, `'SVM'` |
| `USE_DATA_AUGMENTATION` | Image-level augmentation during CNN training |
| `USE_FEATURE_AUGMENTATION` | Feature-space augmentation for classical ML |
| `USE_FEATURE_PREPROCESSING` | Algorithm-specific feature preprocessing pipeline |
| `USE_GRAPHIC_PREPROCESSING` | Hair removal / contrast enhancement (switches data file paths to `preprocessed_*`) |
| `USE_HAIR_REMOVAL` | Requires `USE_GRAPHIC_PREPROCESSING=True` |
| `USE_COLOR_NORMALIZATION` | Reinhard LAB color normalization; requires pre-fitted `COLOR_NORM_STATS_PATH` |
| `USE_FINE_TUNING` | Unfreeze CNN layers for fine-tuning |
| `USE_METADATA` | Append patient metadata (age, location, sex) to CNN features |
| `USE_FOCAL_LOSS` | Replace cross-entropy with focal loss (Lin et al., 2017) |
| `LABEL_SMOOTHING` | Label smoothing coefficient (`0.0` = off, `0.1` recommended) |
| `USE_MIXUP` | Mixup augmentation during CNN batch training (Zhang et al., 2018) |
| `USE_TTA` | Test-Time Augmentation: average `TTA_N_STEPS` augmented predictions |
| `USE_MC_DROPOUT` | Monte Carlo Dropout uncertainty quantification at inference |
| `NUM_KFOLDS` / `NUM_ITERATIONS` | Cross-validation folds and repetitions |

## Architecture

### Data Flow

```
res/train_files.txt  (tab-separated: image_path \t label)
        |
        v
utils/data_loaders.py  ->  MemoryEfficientDataGenerator (CNN) or batch numpy loading (FE)
        |
        +---> pipelines/cnn_classifier.py       (end-to-end CNN training + eval)
        |
        +---> pipelines/feature_extraction.py   (CNN feature extraction -> classical ML)
```

### Two Pipelines

**`cnn_classifier`**: Loads images via `MemoryEfficientDataGenerator`, trains a pretrained CNN (VGG19/Inception/ResNet/Xception/EfficientNet) with optional fine-tuning and class-weighted loss. Saves models and metrics under `results/cnn_classifier_<MODEL>_<flags>/`.

**`feature_extraction`**: Uses the same CNN as a frozen feature extractor (removes the classification head), feeds the extracted feature vectors into a classical ML model. Optionally appends metadata features. Saves under `results/feature_extraction_<MODEL>_<flags>/<classifier>/`.

Result directory names are built automatically from config flags, e.g.:
`results/feature_extraction_ResNet_use_augmentation_use_feature_augmentation_/extratrees/`

### Feature Preprocessing Pipeline

Each classical ML algorithm has a dedicated preprocessing pipeline in `src/preprocessing/feature/algorithm/` (e.g., `extratrees.py`, `xgboost.py`). These extend `AlgorithmPreprocessingPipeline` (abstract base in `src/preprocessing/feature/base/algorithm.py`) and configure steps such as normalization, dimensionality reduction, outlier removal, and class balancing.

The factory `PreprocessingPipelineFactory` (in `src/preprocessing/feature/pipeline.py`) instantiates the correct pipeline for `CLASSICAL_CLASSIFIER_MODEL`. Fitted pipelines are saved with `joblib` and reloaded during inference.

### Graphic Preprocessing Pipeline

`src/preprocessing/graphic/pipeline.py` chains steps (hair removal, contrast enhancement, Reinhard color normalization) and is applied offline before training. When `USE_GRAPHIC_PREPROCESSING=True`, `config.py` automatically redirects data paths to `res/preprocessed_*_files.txt`.

### Analysis / Results Aggregation

`src/analysis/aggregate_results.py` scans the `results/` directory tree, parses `model_performance_summary.csv` and `per_class_metrics.csv` from each experiment folder, and consolidates them into:
- `res/all_models_general.csv` — one row per experiment
- `res/all_models_per_class.csv` — one row per class per experiment

### Key Paths

| Path | Purpose |
|---|---|
| `src/config.py` | All experiment hyperparameters and flags |
| `src/main.py` | CLI entry point |
| `res/` | Dataset split files and aggregated CSVs |
| `results/` | Per-experiment outputs (models, metrics, plots) |
| `results/chimera/best_weights.h5` | Hair removal model weights (must exist before `USE_HAIR_REMOVAL=True`) |
| `res/color_norm_stats.joblib` | Fitted Reinhard stats (must exist before `USE_COLOR_NORMALIZATION=True`) |
| `src/models/cnn_models.py` | CNN architecture loading, focal loss, Grad-CAM |
| `src/models/classical_models.py` | Classical classifier definitions and hyperparameters |
| `src/utils/fold_utils.py` | Saves per-fold metrics to `fold_results_summary.csv` |
| `src/utils/metadata_extractor.py` | Extracts patient metadata features from `res/metadata.csv` |
| `src/utils/calibration.py` | Expected Calibration Error (ECE) computation |
| `src/preprocessing/graphic/color_normalization.py` | Reinhard LAB color normalization step |

## Sub-module Documentation

Detailed CLAUDE.md files exist for each major sub-module:

- [`src/CLAUDE.md`](src/CLAUDE.md) — import conventions, result dir naming
- [`src/models/CLAUDE.md`](src/models/CLAUDE.md) — CNN and classical model details
- [`src/pipelines/CLAUDE.md`](src/pipelines/CLAUDE.md) — training loop internals and output structure
- [`src/preprocessing/feature/CLAUDE.md`](src/preprocessing/feature/CLAUDE.md) — feature preprocessing pipeline
- [`src/preprocessing/graphic/CLAUDE.md`](src/preprocessing/graphic/CLAUDE.md) — image preprocessing and hair removal
- [`src/utils/CLAUDE.md`](src/utils/CLAUDE.md) — data loaders, fold utils, metadata extractor
- [`src/analysis/CLAUDE.md`](src/analysis/CLAUDE.md) — results aggregation and analysis tools

## Prerequisites

Before running with `USE_HAIR_REMOVAL=True`, the hair removal model weights must be
available at `results/chimera/best_weights.h5`. Train it via:

```bash
cd src
python -m preprocessing.graphic.hair_removal.training.trainer
```

Or place pre-trained weights at that path directly.

Before running with `USE_COLOR_NORMALIZATION=True`, fit reference LAB statistics on
training images and save to `res/color_norm_stats.joblib`:

```python
from preprocessing.graphic.color_normalization import compute_reference_stats
import joblib
train_paths = [line.split('\t')[0] for line in open('res/train_files.txt')]
stats = compute_reference_stats(train_paths)
joblib.dump(stats, 'res/color_norm_stats.joblib')
```
