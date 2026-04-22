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
        from config import cfg
        return XGBClassifier(**_XGB_PARAMS, num_class=cfg.num_classes, random_state=random_state)

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


# ---------------------------------------------------------------------------
# Optuna TPE search spaces — one suggest callback per classifier.
# Each callback takes an Optuna trial and returns a dict of classifier__*
# parameters that can be applied via sklearn's Pipeline.set_params().
# ---------------------------------------------------------------------------

def _space_rf(trial):
    return {
        'classifier__n_estimators':      trial.suggest_int('n_estimators', 100, 400, step=50),
        'classifier__max_depth':         trial.suggest_categorical('max_depth', [None, 15, 30, 45]),
        'classifier__min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'classifier__min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 4),
    }


def _space_xgb(trial):
    return {
        'classifier__n_estimators':     trial.suggest_int('n_estimators', 100, 400, step=50),
        'classifier__learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'classifier__max_depth':        trial.suggest_int('max_depth', 3, 8),
        'classifier__subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'classifier__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'classifier__reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
    }


def _space_lgbm(trial):
    return {
        'classifier__n_estimators':      trial.suggest_int('n_estimators', 100, 500, step=50),
        'classifier__learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'classifier__num_leaves':        trial.suggest_int('num_leaves', 31, 127),
        'classifier__min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }


def _space_hgb(trial):
    return {
        'classifier__max_iter':         trial.suggest_int('max_iter', 100, 500, step=50),
        'classifier__learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'classifier__max_leaf_nodes':   trial.suggest_int('max_leaf_nodes', 31, 127),
        'classifier__min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
    }


def _space_et(trial):
    return {
        'classifier__n_estimators':      trial.suggest_int('n_estimators', 100, 400, step=50),
        'classifier__max_depth':         trial.suggest_categorical('max_depth', [None, 20, 40]),
        'classifier__min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'classifier__min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 4),
    }


def _space_svm(trial):
    return {
        'classifier__C':      trial.suggest_float('C', 0.1, 100.0, log=True),
        'classifier__gamma':  trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'classifier__kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
    }


_OPTUNA_SPACES = {
    'RandomForest':         _space_rf,
    'XGBoost':              _space_xgb,
    'LightGBM':             _space_lgbm,
    'HistGradientBoosting': _space_hgb,
    'ExtraTrees':           _space_et,
    'SVM':                  _space_svm,
}


class _OptunaSearchResult:
    """Minimal GridSearchCV-compatible facade around an Optuna study."""
    def __init__(self, best_estimator, best_params, best_score, study):
        self.best_estimator_ = best_estimator
        self.best_params_    = best_params
        self.best_score_     = best_score
        self.study_          = study


def tune_hyperparameters(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    cv: int = 5,
    n_trials: int = 50,
    tune_sample_fraction: float = 0.5,
    class_weights: Optional[Dict[int, float]] = None,
    random_state: int = 42,
) -> _OptunaSearchResult:
    """
    Optuna TPE hyperparameter search on a stratified subsample (§1.7 A2 audit).

    Args:
        pipeline:             Unfitted sklearn Pipeline (from ``create_ml_pipeline``).
        X:                    Full training feature matrix (N, D).
        y:                    Full training labels (N,).
        classifier_name:      Classifier name (resolves the Optuna search space).
        cv:                   Stratified CV folds evaluated per trial.
        n_trials:             Max Optuna trials (default 50).
        tune_sample_fraction: Fraction of training data used for tuning.
        class_weights:        Optional per-class sample weights via ``sample_weight``.
        random_state:         Seed for reproducible TPE sampling and CV splits.

    Returns:
        _OptunaSearchResult with best_estimator_ refit on the subsample.
    """
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    if classifier_name not in _OPTUNA_SPACES:
        raise ValueError(
            f"No Optuna search space registered for '{classifier_name}'. "
            f"Valid: {list(_OPTUNA_SPACES)}"
        )
    suggest = _OPTUNA_SPACES[classifier_name]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=tune_sample_fraction, random_state=random_state
    )
    _, tune_idx = next(splitter.split(X, y))
    X_tune = X[tune_idx]
    y_tune = y[tune_idx]

    sample_weights_full = None
    if class_weights is not None:
        sample_weights_full = np.array([class_weights[label] for label in y_tune])

    print(f"Optuna TPE tuning on {len(X_tune)} samples "
          f"({tune_sample_fraction * 100:.0f}% subsample), n_trials={n_trials}")

    inner_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial):
        params = suggest(trial)
        scores = []
        for train_i, val_i in inner_cv.split(X_tune, y_tune):
            est = clone(pipeline).set_params(**params)
            fit_kwargs = {}
            if sample_weights_full is not None:
                fit_kwargs['classifier__sample_weight'] = sample_weights_full[train_i]
            est.fit(X_tune[train_i], y_tune[train_i], **fit_kwargs)
            pred = est.predict(X_tune[val_i])
            scores.append(balanced_accuracy_score(y_tune[val_i], pred))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {f'classifier__{k}': v for k, v in study.best_params.items()}
    best_estimator = pipeline.set_params(**best_params)
    fit_kwargs = {}
    if sample_weights_full is not None:
        fit_kwargs['classifier__sample_weight'] = sample_weights_full
    best_estimator.fit(X_tune, y_tune, **fit_kwargs)

    print(f"Best params: {study.best_params}")
    print(f"Best CV balanced_accuracy: {study.best_value:.4f}")

    return _OptunaSearchResult(
        best_estimator=best_estimator,
        best_params=study.best_params,
        best_score=study.best_value,
        study=study,
    )


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
