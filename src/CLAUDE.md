# src/CLAUDE.md

This is the working directory for all commands. All modules use bare relative imports
(`from config import ...`), so every command must be run from here.

## Module Map

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for all experiment flags and hyperparameters |
| `main.py` | CLI entry point — parses args, calls pipelines |
| `models/` | CNN architecture loading and classical ML classifiers |
| `pipelines/` | End-to-end training loops (CNN classifier and feature extraction) |
| `preprocessing/` | Image-level (graphic) and feature-level preprocessing |
| `utils/` | Data loading, fold utilities, metadata extraction |
| `analysis/` | Results aggregation across experiments |

## Import Convention

All intra-package imports are bare (no `src.` prefix), e.g.:

```python
from config import CNN_MODEL
from utils.data_loaders import load_paths_labels
from preprocessing.feature.pipeline import apply_feature_preprocessing
```

This works only when the CWD is `src/`. Do not add `src.` prefixes or use relative
dots for top-level imports — it will break.

## Result Directory Naming

Both pipelines auto-construct result directory names from active config flags:

```
results/cnn_classifier_<MODEL>_[hair_removal_][contrast_][use_augmentation_]/
results/feature_extraction_<MODEL>_[use_augmentation_][use_feature_augmentation_]/<classifier>/
```

Example: `results/feature_extraction_ResNet_use_augmentation_/extratrees/`
