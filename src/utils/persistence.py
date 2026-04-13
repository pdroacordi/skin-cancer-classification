"""
Centralized result persistence layer.

Previously, disk writes were scattered across at least 8 call sites in both
pipeline files: open().write(), pd.to_csv(), np.save(), joblib.dump(), etc.
Reading any pipeline required tracing all 3 000+ lines to know what was written.

Now there is a single public function: save_run_results().
No pipeline code outside this module writes result files to disk.

File format decisions:
- CSV over text: every tabular result is a CSV with snake_case headers and
  index=False so pandas can read it without surprises.
- metadata.json over directory-name inference: analysis tools no longer need
  regex to know which flags were active.
- model_performance_summary.csv and per_class_metrics.csv are preserved with
  their original column names so aggregate_results.py keeps working unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from core.run_context import RunContext
from core.types import EvalResult, FoldResult, RunArtifact


def save_run_results(
    ctx: RunContext,
    fold_results: List[FoldResult],
    final_metrics: EvalResult,
    artifacts: List[RunArtifact],
    status: str = "success",
    notes: str = "",
) -> None:
    """
    Write all result files for a completed pipeline run.

    This is the ONLY function permitted to write result files to disk.
    Pipeline code must call this function instead of writing files inline.

    Written files:
        cv_results.csv             — per-fold CV metrics
        fold_results_summary.csv   — same data, legacy name for fold_utils compat
        model_performance_summary.csv — aggregated final-model metrics (legacy format)
        per_class_metrics.csv      — per-class F1/precision/recall (legacy format)
        metadata.json              — machine-readable run record

    Args:
        ctx:           RunContext created at pipeline startup.
        fold_results:  List of FoldResult from cross-validation (may be empty).
        final_metrics: Aggregated EvalResult from the final model(s) on the test set.
        artifacts:     List of RunArtifact describing produced files.
        status:        "success" | "failed" | "partial".
        notes:         Free-text annotation (empty string by default).
    """
    result_dir = ctx.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    artifact_records: List[Dict[str, str]] = []

    # ------------------------------------------------------------------ #
    # 1. CV results (per-fold)                                             #
    # ------------------------------------------------------------------ #
    if fold_results:
        cv_rows = [
            {
                "iteration":      fr.iteration,
                "fold":           fr.fold,
                "accuracy":       fr.eval_result.accuracy,
                "macro_precision":fr.eval_result.macro_precision,
                "macro_recall":   fr.eval_result.macro_recall,
                "macro_f1":       fr.eval_result.macro_f1,
                "macro_auc_roc":  fr.eval_result.macro_auc_roc,
                "ece":            fr.eval_result.ece,
            }
            for fr in fold_results
        ]
        df_cv = pd.DataFrame(cv_rows)

        cv_path = result_dir / "cv_results.csv"
        df_cv.to_csv(cv_path, index=False)
        artifact_records.append({
            "path": "cv_results.csv",
            "description": "Per-fold cross-validation metrics (accuracy, macro-F1, AUC-ROC, ECE)",
        })

        # Legacy name expected by fold_utils and analysis tools
        legacy_fold_path = result_dir / "fold_results_summary.csv"
        df_cv.rename(columns={
            "macro_precision": "precision",
            "macro_recall":    "recall",
            "macro_f1":        "f1_score",
        }).to_csv(legacy_fold_path, index=False)

    # ------------------------------------------------------------------ #
    # 2. Final model metrics — model_performance_summary.csv (legacy fmt) #
    # ------------------------------------------------------------------ #
    summary_row = {
        "model_idx":          1,
        "accuracy":           final_metrics.accuracy,
        "macro_avg_precision":final_metrics.macro_precision,
        "macro_avg_recall":   final_metrics.macro_recall,
        "macro_avg_f1":       final_metrics.macro_f1,
        "macro_auc_roc":      final_metrics.macro_auc_roc,
        "ece":                final_metrics.ece,
    }
    summary_path = result_dir / "model_performance_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    artifact_records.append({
        "path": "model_performance_summary.csv",
        "description": "Aggregated final-model test-set metrics",
    })

    # ------------------------------------------------------------------ #
    # 3. Per-class metrics                                                 #
    # ------------------------------------------------------------------ #
    if final_metrics.per_class:
        per_class_rows = [
            {
                "model_idx":  1,
                "class_name": cls_name,
                "precision":  m["precision"],
                "recall":     m["recall"],
                "f1_score":   m["f1"],
                "support":    m["support"],
            }
            for cls_name, m in final_metrics.per_class.items()
        ]
        per_class_path = result_dir / "per_class_metrics.csv"
        pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
        artifact_records.append({
            "path": "per_class_metrics.csv",
            "description": "Per-class precision, recall, F1-score and support",
        })

    # ------------------------------------------------------------------ #
    # 4. Additional user-provided artifacts                               #
    # ------------------------------------------------------------------ #
    for art in artifacts:
        artifact_records.append({"path": art.path, "description": art.description})

    # ------------------------------------------------------------------ #
    # 5. metadata.json                                                     #
    # ------------------------------------------------------------------ #
    # Round to 4 decimal places for readability; preserve nan as null
    def _safe_round(v: Any) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, float):
            return round(v, 4)
        return v

    summary_metrics = {k: _safe_round(v) for k, v in final_metrics.to_flat_dict().items()}

    ctx.finalize(
        metrics=summary_metrics,
        artifacts=artifact_records,
        status=status,
        notes=notes,
    )


def save_multiple_model_stats(
    model_evals: List[EvalResult],
    result_dir: Path,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Write statistical_analysis.csv and per_class_metrics.csv for multiple
    final models (NUM_FINAL_MODELS > 1).

    This replaces the two large inline stat-computation blocks that existed
    identically in both cnn_classifier.py and feature_extraction.py.

    Args:
        model_evals:  One EvalResult per trained final model.
        result_dir:   Directory where files will be written.
        class_names:  Optional class label list.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Per-model summary rows
    model_rows = []
    class_rows = []

    for idx, ev in enumerate(model_evals, start=1):
        model_rows.append({
            "model_idx":          idx,
            "accuracy":           ev.accuracy,
            "macro_avg_precision":ev.macro_precision,
            "macro_avg_recall":   ev.macro_recall,
            "macro_avg_f1":       ev.macro_f1,
            "macro_auc_roc":      ev.macro_auc_roc,
            "ece":                ev.ece,
        })
        for cls_name, m in ev.per_class.items():
            class_rows.append({
                "model_idx":  idx,
                "class_name": cls_name,
                "precision":  m["precision"],
                "recall":     m["recall"],
                "f1_score":   m["f1"],
                "support":    m["support"],
            })

    df_models = pd.DataFrame(model_rows)
    df_models.to_csv(result_dir / "model_performance_summary.csv", index=False)

    if class_rows:
        pd.DataFrame(class_rows).to_csv(result_dir / "per_class_metrics.csv", index=False)

    # Aggregate statistics with 95% confidence intervals
    metric_arrays = {
        "accuracy":    np.array([r["accuracy"]           for r in model_rows]),
        "macro_f1":    np.array([r["macro_avg_f1"]       for r in model_rows]),
        "precision":   np.array([r["macro_avg_precision"] for r in model_rows]),
        "recall":      np.array([r["macro_avg_recall"]   for r in model_rows]),
        "macro_auc":   np.array([r["macro_auc_roc"]      for r in model_rows]),
    }

    stat_rows = []
    for metric_name, values in metric_arrays.items():
        clean = values[~np.isnan(values)]
        if len(clean) < 2:
            ci_lo, ci_hi = float("nan"), float("nan")
        else:
            ci_lo, ci_hi = stats.t.interval(
                0.95, len(clean) - 1,
                loc=np.mean(clean),
                scale=stats.sem(clean),
            )
        stat_rows.append({
            "metric":  metric_name,
            "mean":    round(float(np.nanmean(values)), 4),
            "std":     round(float(np.nanstd(values)), 4),
            "median":  round(float(np.nanmedian(values)), 4),
            "min":     round(float(np.nanmin(values)), 4),
            "max":     round(float(np.nanmax(values)), 4),
            "ci_95_lo":round(float(ci_lo), 4) if not math.isnan(ci_lo) else None,
            "ci_95_hi":round(float(ci_hi), 4) if not math.isnan(ci_hi) else None,
        })

    pd.DataFrame(stat_rows).to_csv(result_dir / "statistical_analysis.csv", index=False)
