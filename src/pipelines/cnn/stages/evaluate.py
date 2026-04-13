"""
CNN evaluation stage.

Single responsibility: (y_true, y_pred, y_prob) → EvalResult.
Delegates all computation to utils.metrics.compute_classification_metrics
so both pipelines share identical metric definitions.

This is intentionally a thin module.  Its purpose is to give the evaluation
step a named location in the pipeline package, making the stage boundary
explicit even though the logic lives in utils/metrics.py.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.types import EvalResult
from utils.metrics import compute_classification_metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> EvalResult:
    """
    Compute all classification metrics for a set of predictions.

    Pure function — no I/O, no side effects.  Directly testable with
    synthetic numpy arrays.

    Args:
        y_true:      (N,) ground-truth integer class indices.
        y_pred:      (N,) predicted integer class indices.
        y_prob:      (N, C) softmax probabilities.
        class_names: Optional list of C class name strings.

    Returns:
        EvalResult with accuracy, macro-F1, AUC-ROC, ECE, and per-class breakdown.
    """
    return compute_classification_metrics(y_true, y_pred, y_prob, class_names)
