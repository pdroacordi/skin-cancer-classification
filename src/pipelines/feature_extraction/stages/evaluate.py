"""
Feature-extraction pipeline evaluation stage.

Mirrors pipelines/cnn/stages/evaluate.py — both delegate to
utils.metrics.compute_classification_metrics so the metric definitions
are byte-for-byte identical across pipelines.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from core.types import EvalResult
from utils.metrics import compute_classification_metrics


def evaluate_classifier(
    model,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, EvalResult]:
    """
    Run inference with a classical ML classifier and compute all metrics.

    Args:
        model:         Fitted sklearn-compatible classifier.
        test_features: (N_test, D) float32 feature matrix.
        test_labels:   (N_test,) integer ground-truth labels.
        class_names:   Optional class label strings.

    Returns:
        (y_true, y_pred, y_prob, EvalResult)
    """
    y_true = test_labels
    y_pred = model.predict(test_features)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(test_features)
    else:
        n_classes = len(np.unique(y_true))
        y_prob = np.eye(n_classes)[y_pred.astype(int)]

    eval_result = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    return y_true, y_pred, y_prob, eval_result
