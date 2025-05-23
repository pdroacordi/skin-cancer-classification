"""Collapses every text file inside *./results* into two clean pandas DataFrames:
*train_df* and *test_df*.

No model loading, no NumPy arrays in memory – we stick to text‑parsing so the
script executes in a few seconds even on a modest laptop."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd

from .constants import CNN_MODELS, ML_CLASSIFIERS
from .metrics_parser import parse_metrics

@dataclass
class CollectorConfig:
    results_dir: Path = Path("results")


class ResultsCollector:
    """Walks the *results* tree and harvests the numbers we need."""

    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.train_records: List[dict] = []
        self.test_records: List[dict] = []

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def collect(self) -> None:
        self._collect_cnn()
        self._collect_feature_extraction()

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.DataFrame(self.train_records)
        test_df = pd.DataFrame(self.test_records)
        return train_df, test_df

    # ---------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------
    def _collect_cnn(self) -> None:
        for cnn_dir in self.cfg.results_dir.glob("cnn_classifier_*"):
            model_name = next((m for m in CNN_MODELS if m.lower() in cnn_dir.name.lower()), "Unknown")
            # training metrics (cross‑val)
            train_file = cnn_dir / "overall_results.txt"
            if train_file.exists():
                self.train_records.append({
                    "kind": "CNN",
                    "network": model_name,
                    "classifier": "CNN",
                    **parse_metrics(train_file),
                })
            # test metrics
            test_file = cnn_dir / "final_model" / "evaluation_results.txt"
            if test_file.exists():
                self.test_records.append({
                    "kind": "CNN",
                    "network": model_name,
                    "classifier": "CNN",
                    **parse_metrics(test_file),
                })

    def _collect_feature_extraction(self) -> None:
        for fe_dir in self.cfg.results_dir.glob("feature_extraction_*"):
            extractor_name = next((m for m in CNN_MODELS if m.lower() in fe_dir.name.lower()), "Unknown")
            for clf in ML_CLASSIFIERS:
                clf_dir = fe_dir / clf.lower()
                if not clf_dir.exists():
                    continue
                # training metrics ------------------------------------------------
                train_file = clf_dir / "overall_results.txt"
                if train_file.exists():
                    self.train_records.append({
                        "kind": "FE",
                        "network": extractor_name,
                        "classifier": clf,
                        **parse_metrics(train_file),
                    })
                # test metrics -----------------------------------------------------
                test_file = clf_dir / "final_model" / "final_model_test_results.txt"
                if test_file.exists():
                    self.test_records.append({
                        "kind": "FE",
                        "network": extractor_name,
                        "classifier": clf,
                        **parse_metrics(test_file),
                    })