# src/preprocessing/feature/CLAUDE.md

Algorithm-specific preprocessing applied to CNN-extracted feature vectors before
classical ML training. Enabled by `USE_FEATURE_PREPROCESSING=True` in `config.py`.

## Concepts

| Term | Meaning |
|---|---|
| Step | A single, stateful transform (fit + transform). All steps extend `BasePreprocessingStep`. |
| Pipeline | An ordered list of steps plus an optional balancing strategy. Extends `AlgorithmPreprocessingPipeline`. |
| Factory | `PreprocessingPipelineFactory` maps algorithm name → pipeline class. |

## Directory Layout

```
feature/
  base/
    algorithm.py   # AlgorithmPreprocessingPipeline (abstract base; save/load with joblib)
    balancing.py   # BalancingStrategy interface
    step.py        # BasePreprocessingStep interface
  steps/
    balancing.py           # SMOTEBalancing, ClassWeightBalancing, TargetedSMOTEBalancing
    dimensionality_reduction.py  # PCA-based reduction
    normalization.py       # StandardScaler / RobustScaler
    outlier_detection.py   # IsolationForest-based outlier removal
    selection.py           # FeatureSelectionStep (mutual_info / f_score / rfe / importance_based)
    threshold.py           # VarianceThresholdStep
  algorithm/
    configs.py        # ALGORITHM_PIPELINE_CONFIGS — declarative step specs for all 5 algorithms
    configurable.py   # ConfigurablePreprocessingPipeline — builds pipeline from configs.py
  pipeline.py      # PreprocessingPipelineFactory + apply_feature_preprocessing()
```

## Per-Algorithm Pipelines

Each pipeline calls `_configure_pipeline()` in its `__init__` to append steps:

| Algorithm | Steps | Balancing |
|---|---|---|
| ExtraTrees | VarianceThreshold → RobustNorm → MutualInfo(95th pct) | SMOTE-ENN |
| RandomForest | VarianceThreshold → OutlierRemoval → StandardNorm → MutualInfo(95th pct) | SMOTE-ENN |
| XGBoost | VarianceThreshold → OutlierRemoval → StandardNorm → PCA(95%) → MutualInfo(90th pct) | SMOTE-ENN |
| LightGBM | VarianceThreshold → RobustNorm → MutualInfo(90th pct) | ClassWeight |
| HistGradientBoosting | VarianceThreshold → RobustNorm → MutualInfo(90th pct) | ClassWeight |
| SVM | OutlierRemoval → VarianceThreshold → StandardNorm → MutualInfo(60th pct) → PCA(95%) | ClassWeight |

**Available selection steps** (in `steps/selection.py`):
`FeatureSelectionStep` — supports `mutual_info`, `f_score`, `rfe`, `importance_based`.
`CorrelationBasedSelection` has been removed.

## Fit / Transform Contract

- `fit(X, y)` — fits all steps sequentially on training data; outlier removal also
  filters the label array in place during fit.
- `transform(X, y, training=True/False)` — applies steps; outlier removal is
  **skipped on test data** (`training=False`); balancing is **only applied when
  `training=True`**.
- Fitted pipelines are persisted with `joblib` and reloaded for inference so test
  transforms are consistent with training transforms.

## Entry Point

```python
from preprocessing.feature.pipeline import apply_feature_preprocessing

X_proc, y_proc, pipeline = apply_feature_preprocessing(
    features, labels,
    algorithm='ExtraTrees',
    training=True,
    save_path='path/to/pipeline.joblib'
)
```

On inference (`training=False`) the function loads the saved pipeline from
`save_path` instead of fitting a new one.

## Adding a New Algorithm Pipeline

1. Add a new key to `ALGORITHM_PIPELINE_CONFIGS` in `algorithm/configs.py`:
   ```python
   'MyAlgorithm': {
       'steps': [
           ('variance_threshold', {'threshold': 0}),
           ('normalization',      {'method': 'robust'}),
           ('feature_selection',  {'method': 'mutual_info', 'percentile': 90}),
       ],
       'balancing': ('class_weight', {}),
   },
   ```
2. If the step name is new, add its instantiation to `_build_step()` in `algorithm/configurable.py`.
3. No other file changes required — `PreprocessingPipelineFactory` picks it up automatically.
