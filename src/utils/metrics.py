"""
Centralized classification metric computation.

Previously, identical blocks of metric code existed in both cnn_classifier.py
and feature_extraction.py.  Any change (new metric, different AUC variant)
had to be applied twice — and could silently diverge.

This module is the single source of truth.  Both pipelines call
compute_classification_metrics() and receive an EvalResult, guaranteeing
identical metric definitions across experiments.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from core.types import EvalResult
from utils.calibration import expected_calibration_error


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> EvalResult:
    """
    Compute a complete EvalResult from prediction arrays.

    No I/O is performed.  This is a pure function — identical inputs always
    produce identical outputs.  It is directly testable with synthetic arrays
    without any model or filesystem dependency.

    Args:
        y_true:       (N,) integer ground-truth class indices.
        y_pred:       (N,) integer predicted class indices.
        y_prob:       (N, C) softmax probabilities.
        class_names:  Optional list of C class name strings.  When omitted,
                      keys in per_class will be "class_0", "class_1", …

    Returns:
        EvalResult with all scalar metrics and per-class breakdown.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Macro-averaged AUC-ROC (one-vs-rest, standard in medical AI literature).
    # Wrapped in try/except because roc_auc_score requires ≥2 classes in y_true.
    try:
        from tensorflow.keras.utils import to_categorical  # lazy import: avoids TF at import time
        n_classes = y_prob.shape[1]
        y_true_oh = to_categorical(y_true, num_classes=n_classes)
        macro_auc = float(
            roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro")
        )
    except Exception:
        macro_auc = float("nan")

    # Expected Calibration Error (Guo et al., 2017, "On Calibration of Modern
    # Neural Networks").  nan when computation fails (e.g. empty splits).
    try:
        ece = float(expected_calibration_error(y_true, y_prob))
    except Exception:
        ece = float("nan")

    # Build per-class dict.  classification_report uses str(class_index) as keys.
    n_classes = y_prob.shape[1]
    per_class: dict = {}
    for i in range(n_classes):
        key = str(i)
        label = class_names[i] if class_names else f"class_{i}"
        if key in report:
            per_class[label] = {
                "precision": report[key]["precision"],
                "recall":    report[key]["recall"],
                "f1":        report[key]["f1-score"],
                "support":   int(report[key]["support"]),
            }

    return EvalResult(
        accuracy=float(report["accuracy"]),
        macro_precision=float(report["macro avg"]["precision"]),
        macro_recall=float(report["macro avg"]["recall"]),
        macro_f1=float(report["macro avg"]["f1-score"]),
        macro_auc_roc=macro_auc,
        ece=ece,
        per_class=per_class,
        confusion_matrix=cm,
    )


def aggregate_eval_results(results: List[EvalResult]) -> EvalResult:
    """
    Compute mean metrics across a list of EvalResult objects.

    Used to produce an aggregated result from multiple final models
    (NUM_FINAL_MODELS > 1) before writing to metadata.json.

    Returns an EvalResult whose per_class values are also means across
    all provided results.  confusion_matrix is summed (not averaged).
    """
    if not results:
        raise ValueError("Cannot aggregate an empty list of EvalResults.")

    accuracies     = [r.accuracy          for r in results]
    precisions     = [r.macro_precision   for r in results]
    recalls        = [r.macro_recall      for r in results]
    f1s            = [r.macro_f1          for r in results]
    aucs           = [r.macro_auc_roc     for r in results]
    eces           = [r.ece               for r in results]

    # Aggregate per-class metrics across models
    all_class_names = list(results[0].per_class.keys())
    per_class_agg: dict = {}
    for cls_name in all_class_names:
        cls_precisions = [r.per_class[cls_name]["precision"] for r in results if cls_name in r.per_class]
        cls_recalls    = [r.per_class[cls_name]["recall"]    for r in results if cls_name in r.per_class]
        cls_f1s        = [r.per_class[cls_name]["f1"]        for r in results if cls_name in r.per_class]
        cls_supports   = [r.per_class[cls_name]["support"]   for r in results if cls_name in r.per_class]
        per_class_agg[cls_name] = {
            "precision": float(np.mean(cls_precisions)),
            "recall":    float(np.mean(cls_recalls)),
            "f1":        float(np.mean(cls_f1s)),
            "support":   int(cls_supports[0]) if cls_supports else 0,
        }

    # Sum confusion matrices for an aggregate view
    cms = [r.confusion_matrix for r in results if r.confusion_matrix is not None]
    agg_cm = np.sum(cms, axis=0) if cms else None

    return EvalResult(
        accuracy=float(np.mean(accuracies)),
        macro_precision=float(np.mean(precisions)),
        macro_recall=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1s)),
        macro_auc_roc=float(np.nanmean(aucs)),
        ece=float(np.nanmean(eces)),
        per_class=per_class_agg,
        confusion_matrix=agg_cm,
    )
