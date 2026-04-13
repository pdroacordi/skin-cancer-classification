"""
Dynamic Ensemble Selection for the feature extraction pipeline.

Wraps DESlib (0.3.7) algorithms in an sklearn-compatible interface that
integrates cleanly with the project's save/load and evaluation conventions.

Usage::

    from models.dynamic_ensemble import DynamicEnsembleSelector

    # pool_classifiers: list of fitted sklearn-compatible estimators
    des = DynamicEnsembleSelector(algorithm='knorau', k_neighbors=7)
    des.fit(pool_classifiers, X_dsel, y_dsel)
    y_pred = des.predict(X_test)
    y_prob = des.predict_proba(X_test)
    des.save('path/to/des.joblib')

    des2 = DynamicEnsembleSelector.load('path/to/des.joblib')

Algorithm reference:
    knorau      — KNN Oracle Union: for each test point, selects classifiers
                  that are correct on at least one of its K DSEL neighbors.
                  Robust default; no meta-training beyond base classifiers.
    desmi       — DES Multiple Instances: designed for imbalanced multiclass;
                  models each neighborhood as a multi-instance bag and weighs
                  competence by class frequency. Best theoretical fit for
                  HAM10000's severe imbalance (67% nv class).
    metades     — META-DES: learns a meta-classifier to predict which base
                  classifiers are locally competent. Most powerful but requires
                  adequate DSEL coverage per class. Risky for HAM10000 minority
                  classes (df/vasc ~100 total samples → ~30 in DSEL).
    singlebest  — Static selection: picks the globally best classifier on DSEL.
                  No local adaptation; fast, interpretable baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Algorithm registry
# Maps algorithm name → (DESlib class, extra constructor kwargs).
# Comments explain why each default was chosen.
# ---------------------------------------------------------------------------
def _get_algorithm_class(name: str):
    """
    Lazily import and return the DESlib class for *name*.

    Lazy import keeps module-load time fast when DES is not in use.
    """
    if name == 'knorau':
        from deslib.des import KNORAU
        return KNORAU, {}

    if name == 'desmi':
        # DES-MI is designed for imbalanced multiclass: it constructs multi-instance
        # bags around each test point and accounts for class frequency in
        # competence estimation — directly addresses HAM10000's imbalance.
        from deslib.des import DESMI
        return DESMI, {}

    if name == 'metades':
        # META-DES trains a meta-classifier (RandomForest by default in DESlib)
        # on meta-features extracted from the DSEL neighborhood.  More powerful
        # than KNORA-U when DSEL is large, but degrades for tiny minority classes.
        from deslib.des import METADES
        return METADES, {}

    if name == 'singlebest':
        # SingleBest: static, picks the single classifier with highest overall
        # accuracy on DSEL — no local adaptation.  Use as a comparison baseline.
        from deslib.static import SingleBest
        return SingleBest, {}

    valid = ['knorau', 'desmi', 'metades', 'singlebest']
    raise ValueError(
        f"Unknown DES algorithm '{name}'. Choose from {valid}."
    )


# ---------------------------------------------------------------------------
# DynamicEnsembleSelector
# ---------------------------------------------------------------------------

class DynamicEnsembleSelector:
    """
    Sklearn-compatible wrapper around a DESlib dynamic ensemble.

    Parameters
    ----------
    algorithm : str
        One of 'knorau', 'desmi', 'metades', 'singlebest'.
    k_neighbors : int
        Number of DSEL neighbors used to estimate local competence.
        Default is 7 — a multiple of HAM10000's 7 classes, which prevents
        tie-breaking ambiguity in the local accuracy estimate.
    """

    def __init__(self, algorithm: str = 'knorau', k_neighbors: int = 7) -> None:
        self.algorithm   = algorithm
        self.k_neighbors = k_neighbors
        self._des        = None       # DESlib estimator (set after fit)
        self.pool_names: List[str] = []

    # ------------------------------------------------------------------
    # Core sklearn-compatible API
    # ------------------------------------------------------------------

    def fit(
        self,
        pool_classifiers: List[Tuple[str, Any]],
        X_dsel: np.ndarray,
        y_dsel: np.ndarray,
    ) -> "DynamicEnsembleSelector":
        """
        Calibrate the DES on the dynamic selection dataset.

        Parameters
        ----------
        pool_classifiers : list of (name, fitted_estimator) tuples
            Pre-fitted sklearn-compatible classifiers forming the pool.
            Each estimator must implement predict_proba().
        X_dsel : (N, D) float32 array
            Feature matrix of the DSEL split (held-out from classifier training).
        y_dsel : (N,) int array
            Ground-truth labels for X_dsel.

        Returns
        -------
        self
        """
        if not pool_classifiers:
            raise ValueError("pool_classifiers must not be empty.")

        self.pool_names   = [name for name, _ in pool_classifiers]
        fitted_estimators = [clf  for _, clf  in pool_classifiers]

        des_class, extra_kwargs = _get_algorithm_class(self.algorithm)

        # SingleBest does not use k_neighbors; only pass it when relevant.
        if self.algorithm != 'singlebest':
            extra_kwargs = {**extra_kwargs, 'k': self.k_neighbors}

        # DESMI computes N_ = int(n_classifiers * pct_accuracy) internally.
        # With small pools (e.g. 2 classifiers) the default pct_accuracy=0.4
        # yields N_=0 and raises ValueError.  We enforce N_>=1 by computing a
        # safe pct_accuracy floor: max(0.4, 1/n_classifiers).
        if self.algorithm == 'desmi':
            n_classifiers = len(fitted_estimators)
            safe_pct_accuracy = max(0.4, 1.0 / n_classifiers)
            extra_kwargs = {**extra_kwargs, 'pct_accuracy': safe_pct_accuracy}

        self._des = des_class(
            pool_classifiers=fitted_estimators,
            **extra_kwargs,
        )
        self._des.fit(X_dsel.astype(np.float32), y_dsel)

        print(
            f"  DES ({self.algorithm}) fitted on {len(X_dsel)} DSEL samples "
            f"with pool: {self.pool_names}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions via dynamic classifier selection."""
        self._check_fitted()
        return self._des.predict(X.astype(np.float32))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities via dynamic classifier selection.

        DESlib averages probabilities from the locally selected classifiers.
        SingleBest returns probabilities from the single best classifier.
        """
        self._check_fitted()
        return self._des.predict_proba(X.astype(np.float32))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Persist the fitted DES (including pool classifiers) via joblib."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'algorithm':   self.algorithm,
            'k_neighbors': self.k_neighbors,
            'pool_names':  self.pool_names,
            'des':         self._des,
        }, filepath)
        print(f"DynamicEnsembleSelector saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DynamicEnsembleSelector":
        """Load a fitted DES from *filepath*."""
        data = joblib.load(filepath)
        obj              = cls(algorithm=data['algorithm'], k_neighbors=data['k_neighbors'])
        obj.pool_names   = data['pool_names']
        obj._des         = data['des']
        return obj

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        return {
            'algorithm':   self.algorithm,
            'k_neighbors': self.k_neighbors,
            'pool_names':  self.pool_names,
        }

    def _check_fitted(self) -> None:
        if self._des is None:
            raise RuntimeError(
                "DynamicEnsembleSelector has not been fitted yet. "
                "Call fit(pool_classifiers, X_dsel, y_dsel) first."
            )
