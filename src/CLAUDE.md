# src/CLAUDE.md

This is the working directory for all commands. All modules use bare relative imports
(`from config import ...`), so every command must be run from here.

## Module Map

| Module | Responsibility |
|---|---|
| `config.py` | Default values for all experiment flags and hyperparameters; `apply_cli_overrides()` mutates these at runtime |
| `main.py` | CLI entry point — parses all config flags as args, calls `apply_cli_overrides()` before importing pipelines |
| `models/` | CNN architecture loading and classical ML classifiers |
| `pipelines/` | End-to-end training loops (CNN classifier and feature extraction) |
| `preprocessing/` | Image-level (graphic) and feature-level preprocessing |
| `utils/` | Data loading, GPU setup, result naming, fold utilities, metadata extraction |
| `analysis/` | Results aggregation across experiments |
| `tests/` | Pytest test suite — `python -m pytest tests/ -v` from `src/` |

## Import Convention

All intra-package imports are bare (no `src.` prefix), e.g.:

```python
from config import cfg
from utils.data_loaders import load_paths_labels
from preprocessing.feature.pipeline import apply_feature_preprocessing
```

This works only when the CWD is `src/`. Do not add `src.` prefixes or use relative
dots for top-level imports — it will break.

Always import the shared `cfg` object rather than individual config values:

```python
# correct
from config import cfg
batch = cfg.batch_size

# wrong — captures value at import time, ignores CLI overrides
from config import batch_size  # ← this would fail; cfg fields are not module-level names
```

## Result Directory Naming

Both pipelines use `utils/result_naming.py` to build result directory paths from the active config flags:

```
results/cnn_classifier_<MODEL>_[contrast_][hair_removal_][use_augmentation_]/
results/feature_extraction_<MODEL>_[contrast_][hair_removal_][use_augmentation_][use_feature_augmentation_][use_feature_preprocessing_][use_metadata_]/<classifier>/
```

Example: `results/feature_extraction_ResNet_use_augmentation_/extratrees/`

Key functions:
- `cnn_result_dir(base_dir=None)` — base dir for the CNN classifier pipeline
- `feature_extraction_experiment_dir(base_dir=None, cnn_model=None)` — experiment dir (without classifier subdir)
- `feature_extraction_result_dir(base_dir=None, cnn_model=None, classifier=None)` — full path including classifier subdir
