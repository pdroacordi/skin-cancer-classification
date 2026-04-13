"""
Shared data contracts for both CNN and feature-extraction pipelines.

Using dataclasses instead of anonymous dicts means broken contracts
surface as AttributeError at assignment time, not as KeyError at
consumption time deep inside analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class EvalResult:
    """
    Complete evaluation outcome for one model on one dataset split.

    All metric values are floats in [0, 1] except where noted.
    macro_auc_roc and ece may be float('nan') when computation fails
    (e.g. single-class batches, missing probabilities).
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_auc_roc: float   # nan when not computable
    ece: float             # Expected Calibration Error; nan when not computable
    # Keys are class names (or "class_0" if names unavailable).
    # Values: {'precision': float, 'recall': float, 'f1': float, 'support': int}
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    def to_flat_dict(self) -> dict:
        """Return scalar metrics as a flat dict (for CSV serialization)."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "macro_auc_roc": self.macro_auc_roc,
            "ece": self.ece,
        }


@dataclass
class FoldResult:
    """
    Evaluation outcome for a single (iteration, fold) pair.

    Stored per-fold during cross-validation and later aggregated by
    save_run_results() into cv_results.csv.
    """

    iteration: int
    fold: int
    eval_result: EvalResult
    model_path: str = ""

    def to_legacy_dict(self) -> dict:
        """
        Convert to the dict format expected by fold_utils.save_fold_results().
        Maintains backward compatibility with the existing fold persistence layer.
        """
        return {
            "iteration": self.iteration,
            "fold": self.fold,
            "accuracy": self.eval_result.accuracy,
            "macro_avg_precision": self.eval_result.macro_precision,
            "macro_avg_recall": self.eval_result.macro_recall,
            "macro_avg_f1": self.eval_result.macro_f1,
            "macro_auc_roc": self.eval_result.macro_auc_roc,
            "ece": self.eval_result.ece,
            # class_report expected by fold_utils JSON export
            "class_report": {"accuracy": self.eval_result.accuracy},
        }


@dataclass
class RunArtifact:
    """
    Pointer to a file produced by a pipeline run.

    Stored in metadata.json so that an LLM consuming the metadata
    knows where each artifact lives and what it contains, without
    having to open the file.
    """

    path: str          # relative to the run's result_dir
    description: str   # human-readable description of content
