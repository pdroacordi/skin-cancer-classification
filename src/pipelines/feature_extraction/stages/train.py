"""
Classical ML classifier training stage.

Single responsibility: given (train_features, train_labels, val_features,
val_labels, classifier_name), fit and return a trained model with evaluation
metrics on the validation set.

No I/O.  The caller (runner.py) handles saving the model via save_model().
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.types import EvalResult
from models.classical_models import (
    create_ml_pipeline,
    tune_hyperparameters,
)
from utils.metrics import compute_classification_metrics


def train_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    classifier_name: str,
    tune_hyperparams: bool = True,
) -> Tuple[object, EvalResult]:
    """
    Train a classical ML classifier and evaluate on the validation set.

    Args:
        train_features: (N_train, D) float32 feature matrix.
        train_labels:   (N_train,) integer class labels.
        val_features:   (N_val, D) float32 feature matrix.
        val_labels:     (N_val,) integer class labels.
        classifier_name: One of 'RandomForest', 'XGBoost', 'AdaBoost',
                         'ExtraTrees', 'SVM'.
        tune_hyperparams: If True, run GridSearchCV before fitting.

    Returns:
        (fitted_model, EvalResult on validation set)
    """
    print(f"Training {classifier_name}...")

    # float32 avoids implicit upcasting inside sklearn estimators
    X_train = train_features.astype(np.float32)
    X_val   = val_features.astype(np.float32)

    pipeline = create_ml_pipeline(classifier_name=classifier_name)

    if tune_hyperparams:
        print("  Running Optuna TPE search...")
        model = tune_hyperparameters(
            pipeline=pipeline,
            X=X_train,
            y=train_labels,
            classifier_name=classifier_name,
            cv=5,
        ).best_estimator_
    else:
        model = pipeline
        model.fit(X_train, train_labels)

    val_pred = model.predict(X_val)

    # Attempt probability estimates for AUC-ROC and ECE.
    # Not all sklearn estimators support predict_proba; fall back gracefully.
    if hasattr(model, "predict_proba"):
        val_prob = model.predict_proba(X_val)
    else:
        # Build a dummy one-hot-like probability matrix from hard predictions
        n_classes = len(np.unique(train_labels))
        val_prob = np.eye(n_classes)[val_pred.astype(int)]

    eval_result = compute_classification_metrics(
        y_true=val_labels,
        y_pred=val_pred,
        y_prob=val_prob,
    )

    print(f"  Validation — acc={eval_result.accuracy:.4f}  "
          f"macro-F1={eval_result.macro_f1:.4f}")

    return model, eval_result
