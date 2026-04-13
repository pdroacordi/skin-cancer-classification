"""
Classical machine learning model definitions for the feature extraction pipeline.
"""

import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# XGBoost and LightGBM are imported lazily inside get_classifier() to avoid
# paying their import cost when they are not being used.

# ---------------------------------------------------------------------------
# Per-algorithm default hyperparameters
# These are separated from the factory function so they can be read and
# documented independently.  Rationale for each choice is in the comments.
# ---------------------------------------------------------------------------

# RandomForest: 200 trees with sqrt(n_features) split criterion gives a good
# bias-variance trade-off for high-dimensional CNN features.  OOB score is
# enabled as a free validation signal without a dedicated hold-out set.
_RF_PARAMS: Dict[str, Any] = {
    'n_estimators': 200,
    'max_depth': None,       # No depth limit — trees are regularised via min_samples_*
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'class_weight': 'balanced',
    'oob_score': True,
    'n_jobs': -1,
}

# XGBoost: shallow trees (max_depth=4) prevent memorisation of noisy CNN
# features.  hist tree method is required for GPU acceleration.  learning_rate
# 0.05 + 200 rounds provides a conservative-but-effective schedule.
_XGB_PARAMS: Dict[str, Any] = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 1,
    'gamma': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'multi:softprob',
    'tree_method': 'hist',   # CPU/GPU unified code path
    'eval_metric': 'mlogloss',
}

# LightGBM: leaf-wise growth strategy explores deep trees more efficiently than
# XGBoost's level-wise approach, giving better macro-F1 on imbalanced multiclass.
# num_leaves=63 (~2^6-1) provides ample capacity for CNN embedding dimensions
# (~2048) without depth-limit overhead.  Native class_weight avoids SMOTE,
# which can produce noisy synthetic points in high-dimensional embedding spaces.
# verbose=-1 suppresses per-iteration console output.
_LGBM_PARAMS: Dict[str, Any] = {
    'n_estimators':      300,
    'learning_rate':     0.05,
    'num_leaves':        63,      # Tree complexity controller for leaf-wise growth
    'max_depth':         -1,      # Unlimited — num_leaves governs complexity instead
    'min_child_samples': 20,      # Minimum leaf size; guards against tiny minority classes
    'class_weight':      'balanced',
    'objective':         'multiclass',
    'n_jobs':            -1,
    'verbose':           -1,      # Suppress per-tree output
}

# HistGradientBoosting: sklearn-native GBDT with native NaN handling and
# class_weight support (added in sklearn 1.2).  Serves as a dependency-free
# alternative to LightGBM, and as a cross-check since both use histogram-based
# split-finding but differ in tree growth strategy (level-wise vs leaf-wise).
# max_leaf_nodes=63 mirrors LightGBM's num_leaves for comparable capacity.
_HGB_PARAMS: Dict[str, Any] = {
    'max_iter':         300,
    'learning_rate':    0.05,
    'max_leaf_nodes':   63,       # Mirrors LightGBM num_leaves for comparable capacity
    'max_depth':        None,     # Unlimited depth
    'min_samples_leaf': 20,       # Consistent with LightGBM min_child_samples
    'class_weight':     'balanced',
    'early_stopping':   False,    # Explicit iteration control — no hold-out required
}

# ExtraTrees: randomised split thresholds make it faster than RandomForest and
# implicitly more regularised for very high-dimensional inputs.
_ET_PARAMS: Dict[str, Any] = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': -1,
}

# SVM: RBF kernel with C=10 and γ=0.01 was chosen empirically for HAM10000
# CNN features of dimension ~2048.  probability=True enables predict_proba()
# via Platt scaling — this adds overhead but is needed for AUC-ROC.
_SVM_PARAMS: Dict[str, Any] = {
    'C': 10.0,
    'kernel': 'rbf',
    'gamma': 0.01,
    'probability': True,     # Platt scaling — needed for predict_proba / AUC-ROC
    'class_weight': 'balanced',
}


def get_classifier(classifier_name: str, random_state: int = 42) -> Any:
    """
    Instantiate a classifier by name.

    Args:
        classifier_name: One of 'RandomForest', 'XGBoost', 'LightGBM',
                         'HistGradientBoosting', 'ExtraTrees', 'SVM'.
        random_state:    Seed for reproducibility.

    Returns:
        An unfitted sklearn-compatible estimator.

    Raises:
        ValueError: If *classifier_name* is not recognised.
    """
    if classifier_name == "RandomForest":
        return RandomForestClassifier(**_RF_PARAMS, random_state=random_state)

    if classifier_name == "XGBoost":
        from xgboost import XGBClassifier
        from config import NUM_CLASSES  # lazy import — avoids TF/config side-effects at module load
        return XGBClassifier(**_XGB_PARAMS, num_class=NUM_CLASSES, random_state=random_state)

    if classifier_name == "LightGBM":
        from lightgbm import LGBMClassifier  # lazy import — avoids cost when not in use
        return LGBMClassifier(**_LGBM_PARAMS, random_state=random_state)

    if classifier_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(**_HGB_PARAMS, random_state=random_state)

    if classifier_name == "ExtraTrees":
        return ExtraTreesClassifier(**_ET_PARAMS, random_state=random_state)

    if classifier_name == "SVM":
        return SVC(**_SVM_PARAMS, random_state=random_state)

    raise ValueError(
        f"Unsupported classifier: '{classifier_name}'. "
        "Choose from 'RandomForest', 'XGBoost', 'LightGBM', "
        "'HistGradientBoosting', 'ExtraTrees', 'SVM'."
    )


def create_ml_pipeline(classifier_name: str, random_state: int = 42) -> Pipeline:
    """
    Wrap a classifier in an sklearn Pipeline.

    The pipeline currently contains only the classifier step.  Preprocessing
    steps (normalization, feature selection) are handled by the separate
    ``preprocessing.feature`` pipeline so that fitted transformers can be
    persisted and re-applied at inference time.

    Args:
        classifier_name: Name of the classifier (see ``get_classifier``).
        random_state:    Seed for reproducibility.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.
    """
    classifier = get_classifier(classifier_name, random_state=random_state)
    return Pipeline([('classifier', classifier)])


def tune_hyperparameters(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, Any],
    cv: int = 5,
    n_jobs: int = -1,
    tune_sample_fraction: float = 0.5,
    verbose: int = 1,
    class_weights: Optional[Dict[int, float]] = None,
) -> Any:
    """
    Run ``GridSearchCV`` on a stratified subsample.

    Tuning on the full dataset is prohibitively slow for large feature
    matrices, so we draw a stratified subsample of size
    ``tune_sample_fraction * N`` and search on that.

    Args:
        pipeline:             Unfitted sklearn Pipeline (from ``create_ml_pipeline``).
        X:                    Full training feature matrix (N, D).
        y:                    Full training labels (N,).
        param_grid:           GridSearchCV parameter grid.
        cv:                   Number of CV folds inside GridSearchCV.
        n_jobs:               Parallel workers for GridSearchCV.
        tune_sample_fraction: Fraction of training data to use for the
                              hyperparameter search (default 0.5).
        verbose:              GridSearchCV verbosity level.
        class_weights:        Optional per-class sample weights to pass to
                              the classifier's ``fit`` via ``sample_weight``.

    Returns:
        Fitted ``GridSearchCV`` object; best estimator is ``.best_estimator_``.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

    # Draw a stratified subsample so the class distribution is preserved.
    # StratifiedShuffleSplit's `test_size` here is the fraction we *want*,
    # and we take the "test" split indices as our tuning subsample.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=tune_sample_fraction, random_state=42)
    _, tune_idx = next(splitter.split(X, y))
    X_tune = X[tune_idx]
    y_tune = y[tune_idx]

    print(f"Using {len(X_tune)} samples ({tune_sample_fraction * 100:.0f}%) for hyperparameter tuning")

    fit_params: Dict[str, Any] = {}
    if class_weights is not None:
        sample_weights = np.array([class_weights[label] for label in y_tune])
        fit_params['classifier__sample_weight'] = sample_weights
        print("Using class weights during hyperparameter tuning")

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring='balanced_accuracy',
        return_train_score=True,
    )
    grid_search.fit(X_tune, y_tune, **fit_params)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score:   {grid_search.best_score_:.4f}")

    return grid_search


def get_default_param_grid(classifier_name: str) -> Dict[str, Any]:
    """
    Return the default hyperparameter grid for ``tune_hyperparameters``.

    Args:
        classifier_name: Name of the classifier.

    Returns:
        Dict suitable for ``GridSearchCV(param_grid=...)``.

    Raises:
        ValueError: If *classifier_name* is not recognised.
    """
    grids: Dict[str, Dict[str, Any]] = {
        "RandomForest": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 15, 30],
            'classifier__min_samples_split': [2, 10],
            'classifier__min_samples_leaf': [1, 4],
        },
        "XGBoost": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.7, 0.8, 0.9],
            'classifier__colsample_bytree': [0.7, 0.8, 0.9],
        },
        "LightGBM": {
            'classifier__n_estimators':      [100, 200, 300],
            'classifier__learning_rate':     [0.01, 0.05, 0.1],
            'classifier__num_leaves':        [31, 63, 127],
            'classifier__min_child_samples': [10, 20, 50],
        },
        "HistGradientBoosting": {
            'classifier__max_iter':         [100, 200, 300],
            'classifier__learning_rate':    [0.01, 0.05, 0.1],
            'classifier__max_leaf_nodes':   [31, 63, 127],
            'classifier__min_samples_leaf': [10, 20, 50],
        },
        "ExtraTrees": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
        },
        "SVM": {
            'classifier__C': [0.1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'classifier__kernel': ['rbf', 'linear', 'poly'],
        },
    }

    if classifier_name not in grids:
        raise ValueError(
            f"Unsupported classifier: '{classifier_name}'. "
            f"Valid: {list(grids)}"
        )
    return grids[classifier_name]


def save_model(model: Any, save_path: str) -> None:
    """Persist a fitted model to *save_path* using joblib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")


def load_model(load_path: str) -> Any:
    """Load a joblib-persisted model from *load_path*."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    model = joblib.load(load_path)
    print(f"Model loaded from: {load_path}")
    return model
