# src/analysis/CLAUDE.md

Tools for aggregating and analysing experiment results after training runs.

## aggregate_results.py

Scans the entire `results/` directory tree and consolidates metrics from every
experiment into two CSVs under `res/`:

| Output file | Grain | Metric prefix |
|---|---|---|
| `all_models_general.csv` | One row per experiment | `g_` (e.g. `g_accuracy`) |
| `all_models_per_class.csv` | One row per class per experiment | no prefix |

**How it works:**
1. Walks every subdirectory in `results/`.
2. Parses `model_performance_summary.csv` (overall metrics) and
   `per_class_metrics.csv` (per-class metrics) when found.
3. Infers experiment metadata (CNN backbone, kind, algorithm, flags) from the
   directory path using `_infer_meta_from_path()`.
4. Writes both CSVs with consistent columns for downstream analysis.

**Run:**
```bash
# From src/
python -m analysis.aggregate_results          # quiet
python -m analysis.aggregate_results -v       # verbose
```

**Columns inferred from path:**
- `net`: resnet / inception / vgg19 / xception
- `kind`: `cnn_classifier` or `feature_extraction`
- `algorithm`: adaboost / extratrees / randomforest / xgboost / svm (empty for CNN classifier)
- `feature_augmentation`, `data_augmentation`, `hair_removal`, `segmentation`: boolean flags

## plotter.py

Plotting utilities for generating figures from aggregated CSVs (confusion matrices,
per-class bar charts, etc.). Import and call from notebooks or scripts as needed.

## stat_tests.py

Statistical significance tests (e.g. Wilcoxon, Friedman) to compare model
performance distributions across experiments.

## constants.py

Shared constants used across the analysis module (class names, label mappings, etc.).
