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
    algorithm.py   # AlgorithmPreprocessingPipeline (abstract base)
    balancing.py   # BalancingStrategy interface
    step.py        # BasePreprocessingStep interface
  steps/
    balancing.py           # SMOTEBalancing, ClassWeightBalancing, TargetedSMOTEBalancing
    dimensionality_reduction.py  # PCA-based reduction
    normalization.py       # StandardScaler / RobustScaler / MinMaxScaler
    outlier_detection.py   # IsolationForest-based outlier removal
    selection.py           # FeatureSelectionStep (mutual_info / f_score / rfe / importance_based)
                           # CorrelationBasedSelection
    threshold.py           # VarianceThresholdStep
  algorithm/
    adaboost.py    # AdaBoostPipeline
    extratrees.py  # ExtraTreesPipeline
    randomforest.py# RandomForestPipeline
    svm.py         # SVMPipeline
    xgboost.py     # XGBoostPipeline
  pipeline.py      # PreprocessingPipelineFactory + apply_feature_preprocessing()
```

## Per-Algorithm Pipelines

Each pipeline calls `_configure_pipeline()` in its `__init__` to append steps:

| Algorithm | Steps | Balancing |
|---|---|---|
| ExtraTrees | VarianceThreshold → RobustNorm → MutualInfo(95th pct) | ClassWeight |
| RandomForest | VarianceThreshold → RobustNorm → MutualInfo(90th pct) | ClassWeight |
| XGBoost | VarianceThreshold → RobustNorm → MutualInfo(85th pct) | ClassWeight |
| AdaBoost | OutlierRemoval → VarianceThreshold → F-score(85th pct) | ClassWeight |
| SVM | VarianceThreshold → StandardNorm → PCA(95% var) | ClassWeight |

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

1. Create `algorithm/<name>.py` subclassing `AlgorithmPreprocessingPipeline`.
2. Implement `_configure_pipeline()` to populate `self.steps` and optionally
   `self.balancing_strategy`.
3. Register in `PreprocessingPipelineFactory._pipelines` in `pipeline.py`.
4. Add it to `PIPELINE_CLASS_MAP` (auto-derived from the dict, no manual step needed).
