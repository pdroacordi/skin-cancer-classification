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
    knorau-ece  — H3: ECE-guided KNORAU. Competence weighted by
                  (1 − ECE_local) × accuracy_local in the K-neighborhood.
                  Reduces ensemble ECE without sacrificing macro-F1.
    knorau-temp — H3 critical control: standard KNORAU with post-hoc
                  temperature scaling on the ensemble output. Isolates
                  whether ECE gain in knorau-ece is from selection-aware
                  calibration vs. mere post-hoc rescaling.
    conformal-targeted      — H16: CT-DES. Competence estimated only over
                  DSEL neighbors whose true label lies in the conformal
                  prediction set S(x). Targets the specific discrimination
                  problem posed by each test sample.
    conformal-targeted-ece  — H16 + H3 layered: CT-DES with ECE-weighting
                  of filtered competence. Full KNORAU → ECE-DES → CT-DES →
                  CT-DES+ECE ablation condition.
    conformal-targeted-random — H16 kill control: CT-DES with random class
                  sets of matched cardinality |S(x)| instead of APS sets.
                  Isolates targeting claim from filter-regularisation claim.
    desmi       — DES Multiple Instances: designed for imbalanced multiclass;
                  models each neighborhood as a multi-instance bag and weighs
                  competence by class frequency. Best theoretical fit for
                  HAM10000's severe imbalance (67% nv class).
    metades     — META-DES: learns a meta-classifier to predict which base
                  classifiers are locally competent. Most powerful but requires
                  adequate DSEL coverage per class.
    singlebest  — Static selection: picks the globally best classifier on DSEL.
                  No local adaptation; fast, interpretable baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def _get_algorithm_class(name: str):
    """
    Lazily import and return the DESlib class (or custom subclass) for *name*.
    """
    if name == 'knorau':
        from deslib.des import KNORAU
        return KNORAU, {}

    if name == 'knorau-ece':
        return CalibrationAwareKNORAU, {}

    if name == 'knorau-temp':
        # Handled separately in DynamicEnsembleSelector.fit — plain KNORAU
        # with a TemperatureScaledDES wrapper added after fit.
        from deslib.des import KNORAU
        return KNORAU, {'_wrap_temperature': True}

    if name == 'conformal-targeted':
        return ConformalTargetedKNORAU, {'random_kill_control': False}

    if name == 'conformal-targeted-ece':
        return ConformalTargetedECEKNORAU, {}

    if name == 'conformal-targeted-random':
        return ConformalTargetedKNORAU, {'random_kill_control': True}

    if name == 'desmi':
        from deslib.des import DESMI
        return DESMI, {}

    if name == 'metades':
        from deslib.des import METADES
        return METADES, {}

    if name == 'singlebest':
        from deslib.static import SingleBest
        return SingleBest, {}

    valid = [
        'knorau', 'knorau-ece', 'knorau-temp',
        'conformal-targeted', 'conformal-targeted-ece', 'conformal-targeted-random',
        'desmi', 'metades', 'singlebest',
    ]
    raise ValueError(f"Unknown DES algorithm '{name}'. Choose from {valid}.")


# ---------------------------------------------------------------------------
# DynamicEnsembleSelector — public sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class DynamicEnsembleSelector:
    """
    Sklearn-compatible wrapper around a DESlib dynamic ensemble.

    Parameters
    ----------
    algorithm : str
        DES algorithm name (see module docstring for options).
    k_neighbors : int
        Number of DSEL neighbors used to estimate local competence.
        Default is 7 — a multiple of HAM10000's 7 classes.
    """

    def __init__(self, algorithm: str = 'knorau', k_neighbors: int = 7) -> None:
        self.algorithm   = algorithm
        self.k_neighbors = k_neighbors
        self._des        = None
        self.pool_names: List[str] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        pool_classifiers: List[Tuple[str, Any]],
        X_dsel: np.ndarray,
        y_dsel: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "DynamicEnsembleSelector":
        """
        Calibrate the DES on the dynamic selection dataset.

        Parameters
        ----------
        pool_classifiers : list of (name, fitted_estimator) tuples
        X_dsel : (N, D) float32 — features for DES competence estimation.
        y_dsel : (N,) int — ground-truth labels for X_dsel.
        X_val  : (M, D) float32 — validation features. Required when
                 algorithm='knorau-temp' (needed to fit temperature scaler).
        y_val  : (M,) int — validation labels (paired with X_val).
        """
        if not pool_classifiers:
            raise ValueError("pool_classifiers must not be empty.")

        self.pool_names   = [name for name, _ in pool_classifiers]
        fitted_estimators = [clf  for _, clf  in pool_classifiers]

        des_class, extra_kwargs = _get_algorithm_class(self.algorithm)

        wrap_temperature = extra_kwargs.pop('_wrap_temperature', False)

        if self.algorithm not in ('singlebest',):
            extra_kwargs = {**extra_kwargs, 'k': self.k_neighbors}

        if self.algorithm == 'desmi':
            n_classifiers = len(fitted_estimators)
            safe_pct_accuracy = max(0.4, 1.0 / n_classifiers)
            extra_kwargs = {**extra_kwargs, 'pct_accuracy': safe_pct_accuracy}

        self._des = des_class(
            pool_classifiers=fitted_estimators,
            **extra_kwargs,
        )
        self._des.fit(X_dsel.astype(np.float32), y_dsel)

        # For CalibrationAwareKNORAU: precompute DSEL probability cache
        if isinstance(self._des, CalibrationAwareKNORAU):
            self._des.precompute_dsel_probas()

        if wrap_temperature:
            if X_val is None or y_val is None:
                raise ValueError(
                    "algorithm='knorau-temp' requires X_val and y_val to fit "
                    "the temperature scaler."
                )
            val_proba = self._des.predict_proba(X_val.astype(np.float32))
            from utils.calibration import TemperatureScaler
            scaler = TemperatureScaler()
            # Approximate logits as log(p) for temperature fitting
            logits_val = np.log(np.clip(val_proba, 1e-9, 1.0))
            scaler.fit(logits_val, y_val)
            self._des = TemperatureScaledDES(self._des, scaler)

        print(
            f"  DES ({self.algorithm}) fitted on {len(X_dsel)} DSEL samples "
            f"with pool: {self.pool_names}"
        )
        return self

    def set_conformal_sets(self, S: np.ndarray) -> None:
        """
        Provide the conformal prediction sets for the upcoming predict() call.

        Required before predict() / predict_proba() for conformal-targeted
        algorithms.  S is generated by APSConformalPredictor.predict_set().

        Parameters
        ----------
        S : (n_test, n_classes) bool array — S[i, c] = True iff class c is
            in the APS prediction set for test sample i.
        """
        self._check_fitted()
        if not isinstance(self._des, (ConformalTargetedKNORAU,
                                       ConformalTargetedECEKNORAU)):
            raise TypeError(
                f"set_conformal_sets() is only valid for conformal-targeted "
                f"algorithms; got '{self.algorithm}'."
            )
        self._des.set_conformal_sets(S)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._des.predict(X.astype(np.float32))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._des.predict_proba(X.astype(np.float32))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump({
            'algorithm':   self.algorithm,
            'k_neighbors': self.k_neighbors,
            'pool_names':  self.pool_names,
            'des':         self._des,
        }, filepath)
        print(f"DynamicEnsembleSelector saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DynamicEnsembleSelector":
        data = joblib.load(filepath)
        obj              = cls(algorithm=data['algorithm'],
                               k_neighbors=data['k_neighbors'])
        obj.pool_names   = data['pool_names']
        obj._des         = data['des']
        return obj

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
                "Call fit() first."
            )


# ---------------------------------------------------------------------------
# H3 — CalibrationAwareKNORAU
# ---------------------------------------------------------------------------

class CalibrationAwareKNORAU:
    """
    ECE-weighted DES competence for H3 (ECE-Guided Dynamic Ensemble Selection).

    Wraps DESlib's KNORAU with a modified competence function:
        competence(c, x) = (1 − ECE_local(c, N_k(x))) × accuracy_local(c, N_k(x))

    where ECE_local is computed on the K nearest DSEL neighbors using 5 bins
    (fewer than the global 15 to avoid empty bins at K=7).

    Usage::
        clf = CalibrationAwareKNORAU(pool_classifiers=pool, k=7)
        clf.fit(X_dsel, y_dsel)
        clf.precompute_dsel_probas()   # must call after fit
        y_pred = clf.predict(X_test)

    Wrapping in DynamicEnsembleSelector handles precompute_dsel_probas()
    automatically.
    """

    def __init__(self, pool_classifiers=None, k: int = 7, **kwargs) -> None:
        from deslib.des import KNORAU
        self._knorau = KNORAU(pool_classifiers=pool_classifiers, k=k, **kwargs)
        self.k = k
        # (n_dsel, n_classifiers, n_classes) — filled by precompute_dsel_probas()
        self._dsel_proba_cache: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibrationAwareKNORAU":
        self._knorau.fit(X, y)
        return self

    def precompute_dsel_probas(self) -> None:
        """
        Precompute per-classifier softmax probabilities on the DSEL set.

        Cached as (n_dsel, n_classifiers, n_classes) float32 array.
        Must be called after fit() and before predict().
        """
        n_dsel = len(self._knorau.DSEL_data_)
        n_clf  = self._knorau.n_classifiers_
        # Probe number of classes from first classifier
        n_cls  = len(self._knorau.classes_)

        cache = np.zeros((n_dsel, n_clf, n_cls), dtype=np.float32)
        X_dsel = self._knorau.DSEL_data_.astype(np.float32)
        for c, clf in enumerate(self._knorau.pool_classifiers_):
            cache[:, c, :] = clf.predict_proba(X_dsel).astype(np.float32)
        self._dsel_proba_cache = cache

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_batch(X, proba=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._predict_batch(X, proba=True)

    def _predict_batch(self, X: np.ndarray, proba: bool) -> np.ndarray:
        from deslib.util.instance_hardness import hardness_region_competence
        from deslib.base import BaseDS

        # Use KNORAU's kNN to find competence regions
        knorau = self._knorau
        distances, competence_region = knorau.get_competence_region(X)

        competences = self._estimate_competence(competence_region, distances)

        # Select classifiers using KNORAU's select logic
        selected = knorau.select(competences)  # (n, n_clf) bool

        n_samples  = X.shape[0]
        n_clf      = knorau.n_classifiers_
        n_cls      = len(knorau.classes_)

        if proba:
            pool_probas = np.stack(
                [clf.predict_proba(X) for clf in knorau.pool_classifiers_], axis=0
            )                                                      # (n_clf, n, n_cls)
            weights = (selected * competences).T                  # (n_clf, n)
            output  = np.einsum('cn,cnk->nk', weights, pool_probas)  # (n, n_cls)
            totals  = output.sum(axis=1, keepdims=True)
            zero_rows = (totals.squeeze() == 0)
            if zero_rows.any():
                output[zero_rows] = pool_probas[:, zero_rows, :].mean(axis=0)
                totals[zero_rows] = 1.0
            return (output / totals).astype(np.float32)
        else:
            votes = np.zeros((n_samples, n_cls), dtype=np.float64)
            base_preds = np.column_stack([
                clf.predict(X) for clf in knorau.pool_classifiers_
            ])                                                     # (n, n_clf)
            for i in range(n_samples):
                for c in range(n_clf):
                    if selected[i, c]:
                        cls_idx = np.searchsorted(knorau.classes_,
                                                  base_preds[i, c])
                        votes[i, cls_idx] += competences[i, c]
            zero_rows = ~votes.any(axis=1)
            if zero_rows.any():
                votes[zero_rows] = 1.0
            return knorau.classes_[np.argmax(votes, axis=1)]

    def _estimate_competence(
        self,
        competence_region: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        ECE-weighted competence: (1 − ECE_local) × accuracy_local.

        Parameters
        ----------
        competence_region : (n_samples, K) int — DSEL neighbor indices.

        Returns
        -------
        competences : (n_samples, n_classifiers) float
        """
        from utils.calibration import ece_per_neighbors

        if self._dsel_proba_cache is None:
            raise RuntimeError("Call precompute_dsel_probas() after fit().")

        knorau   = self._knorau
        n_samples, K = competence_region.shape
        n_clf    = knorau.n_classifiers_

        competences = np.zeros((n_samples, n_clf), dtype=np.float64)

        for i in range(n_samples):
            neighbors = competence_region[i]                      # (K,)
            correct_mat = knorau.DSEL_processed_[neighbors, :]   # (K, n_clf)
            for c in range(n_clf):
                acc_local = float(correct_mat[:, c].mean())
                probs_local = self._dsel_proba_cache[neighbors, c, :]  # (K, n_cls)
                ece_local   = ece_per_neighbors(probs_local,
                                                correct_mat[:, c],
                                                n_bins=5)
                competences[i, c] = (1.0 - ece_local) * acc_local

        return competences


# ---------------------------------------------------------------------------
# H3 control — TemperatureScaledDES
# ---------------------------------------------------------------------------

class TemperatureScaledDES:
    """
    Post-hoc temperature scaling applied to a fitted DES ensemble output.

    Critical control for H3: if temperature-scaled KNORAU achieves the same
    ECE reduction as CalibrationAwareKNORAU, the selection-aware calibration
    story collapses — the gain is merely from post-hoc rescaling, not from
    competence-weighted selection.

    Wraps any fitted DES that exposes predict() / predict_proba().
    """

    def __init__(self, des, scaler) -> None:
        self._des    = des
        self._scaler = scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Argmax is invariant to temperature scaling, so just forward.
        return self._des.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_proba  = self._des.predict_proba(X)
        logits     = np.log(np.clip(raw_proba, 1e-9, 1.0))
        return self._scaler.transform(logits)


# ---------------------------------------------------------------------------
# H16 — ConformalTargetedKNORAU
# ---------------------------------------------------------------------------

class ConformalTargetedKNORAU:
    """
    CT-DES: competence estimated over DSEL neighbors whose true label is in
    the APS conformal prediction set S(x) of the test sample.

    For test sample x with conformal set S_x:
      1. Find K=7 nearest DSEL neighbors via kNN.
      2. Keep only neighbors n where y_n ∈ S_x (targeted filtering).
      3. If |filtered| < 2 or |S_x| ∈ {1, n_classes}: fall back to KNORAU.
      4. competence(c, x) = accuracy of c on the filtered neighbors.

    Parameters
    ----------
    random_kill_control : bool
        When True, each S_x row is replaced with a random class set of
        matched cardinality at predict time.  This is the experimental kill
        control for H16: if the random version achieves comparable macro-F1,
        the targeting claim is invalidated (gain is filter regularisation,
        not class-targeted selection).
    """

    def __init__(self, pool_classifiers=None, k: int = 7,
                 random_kill_control: bool = False, **kwargs) -> None:
        from deslib.des import KNORAU
        self._knorau            = KNORAU(pool_classifiers=pool_classifiers,
                                         k=k, **kwargs)
        self.k                  = k
        self.random_kill_control = random_kill_control
        self._S: Optional[np.ndarray] = None  # (n_test, n_classes) bool

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalTargetedKNORAU":
        self._knorau.fit(X, y)
        return self

    def set_conformal_sets(self, S: np.ndarray) -> None:
        """
        Provide APS conformal sets for the next predict() / predict_proba() call.

        Parameters
        ----------
        S : (n_test, n_classes) bool — output of APSConformalPredictor.predict_set().
            If random_kill_control=True, each row is silently replaced with a
            random set of matched cardinality at predict time.
        """
        if self.random_kill_control:
            S = _randomize_conformal_sets(S)
        self._S = np.asarray(S, dtype=bool)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_batch(X, proba=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._predict_batch(X, proba=True)

    def _predict_batch(self, X: np.ndarray, proba: bool) -> np.ndarray:
        if self._S is None:
            raise RuntimeError(
                "Call set_conformal_sets(S) before predict() / predict_proba()."
            )
        if len(self._S) != len(X):
            raise ValueError(
                f"S has {len(self._S)} rows but X has {len(X)} samples."
            )

        knorau = self._knorau
        distances, competence_region = knorau.get_competence_region(X)
        competences = self._estimate_competence(competence_region, distances)
        selected    = knorau.select(competences)

        n_samples = X.shape[0]
        n_cls     = len(knorau.classes_)
        n_clf     = knorau.n_classifiers_

        if proba:
            # Pool probabilities: (n_clf, n, n_cls)
            pool_probas = np.stack(
                [clf.predict_proba(X) for clf in knorau.pool_classifiers_], axis=0
            )
            # Weights: competence × selected, shape (n_clf, n)
            weights = (selected * competences).T                  # (n_clf, n)
            output  = np.einsum('cn,cnk->nk', weights, pool_probas)  # (n, n_cls)
            totals  = output.sum(axis=1, keepdims=True)
            # Fallback to uniform average when all competences are zero
            zero_rows = (totals.squeeze() == 0)
            if zero_rows.any():
                output[zero_rows] = pool_probas[:, zero_rows, :].mean(axis=0)
                totals[zero_rows] = 1.0
            return (output / totals).astype(np.float32)
        else:
            votes = np.zeros((n_samples, n_cls), dtype=np.float64)
            base_preds = np.column_stack([
                clf.predict(X) for clf in knorau.pool_classifiers_
            ])
            for i in range(n_samples):
                for c in range(n_clf):
                    if selected[i, c]:
                        cls_idx = np.searchsorted(knorau.classes_,
                                                  base_preds[i, c])
                        votes[i, cls_idx] += competences[i, c]
            zero_rows = ~votes.any(axis=1)
            if zero_rows.any():
                votes[zero_rows] = 1.0
            return knorau.classes_[np.argmax(votes, axis=1)]

    def _estimate_competence(
        self,
        competence_region: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        CT-DES competence: accuracy on S(x)-filtered DSEL neighbors.
        """
        knorau    = self._knorau
        n_samples, K = competence_region.shape
        n_clf     = knorau.n_classifiers_
        n_cls     = len(knorau.classes_)

        competences = np.zeros((n_samples, n_clf), dtype=np.float64)

        for i in range(n_samples):
            S_i = self._S[i]                                      # (n_classes,) bool
            n_in_S = int(S_i.sum())

            neighbors = competence_region[i]                      # (K,)

            # Fallback condition: trivial set or all-classes set
            if n_in_S <= 1 or n_in_S >= n_cls:
                competences[i] = knorau.DSEL_processed_[
                    neighbors, :].sum(axis=0).astype(np.float64)
                continue

            # Keep neighbors whose true label ∈ S_i
            neighbor_labels  = knorau.DSEL_target_[neighbors]    # (K,)
            keep_mask        = S_i[neighbor_labels]               # (K,) bool

            if keep_mask.sum() < 2:
                # Not enough filtered neighbors — fall back to plain KNORAU
                competences[i] = knorau.DSEL_processed_[
                    neighbors, :].sum(axis=0).astype(np.float64)
                continue

            filtered = neighbors[keep_mask]                       # (≤K,) filtered indices
            competences[i] = knorau.DSEL_processed_[
                filtered, :].sum(axis=0).astype(np.float64)

        return competences


# ---------------------------------------------------------------------------
# H16 + H3 layered — ConformalTargetedECEKNORAU
# ---------------------------------------------------------------------------

class ConformalTargetedECEKNORAU(ConformalTargetedKNORAU):
    """
    CT-DES + ECE-weighting: (1 − ECE_local_filtered) × accuracy_local_filtered.

    Combines CT-DES (H16) filtering with ECE-weighting (H3). Serves as the
    fourth ablation condition in the layered experiment:
        KNORAU → ECE-DES → CT-DES → CT-DES+ECE

    Requires precompute_dsel_probas() to be called after fit().
    """

    def __init__(self, pool_classifiers=None, k: int = 7, **kwargs) -> None:
        super().__init__(pool_classifiers=pool_classifiers, k=k,
                         random_kill_control=False, **kwargs)
        self._dsel_proba_cache: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalTargetedECEKNORAU":
        super().fit(X, y)
        return self

    def precompute_dsel_probas(self) -> None:
        """Precompute DSEL probability cache (same logic as CalibrationAwareKNORAU)."""
        knorau = self._knorau
        n_dsel = len(knorau.DSEL_data_)
        n_clf  = knorau.n_classifiers_
        n_cls  = len(knorau.classes_)
        cache  = np.zeros((n_dsel, n_clf, n_cls), dtype=np.float32)
        X_dsel = knorau.DSEL_data_.astype(np.float32)
        for c, clf in enumerate(knorau.pool_classifiers_):
            cache[:, c, :] = clf.predict_proba(X_dsel).astype(np.float32)
        self._dsel_proba_cache = cache

    def _estimate_competence(
        self,
        competence_region: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """CT-DES + ECE: (1 − ECE_local_filtered) × accuracy_local_filtered."""
        from utils.calibration import ece_per_neighbors

        if self._dsel_proba_cache is None:
            raise RuntimeError("Call precompute_dsel_probas() after fit().")

        knorau    = self._knorau
        n_samples, K = competence_region.shape
        n_clf     = knorau.n_classifiers_
        n_cls     = len(knorau.classes_)

        competences = np.zeros((n_samples, n_clf), dtype=np.float64)

        for i in range(n_samples):
            S_i    = self._S[i]
            n_in_S = int(S_i.sum())
            neighbors = competence_region[i]

            if n_in_S <= 1 or n_in_S >= n_cls:
                # KNORAU fallback (no ECE-weighting for trivial sets)
                competences[i] = knorau.DSEL_processed_[
                    neighbors, :].sum(axis=0).astype(np.float64)
                continue

            neighbor_labels = knorau.DSEL_target_[neighbors]
            keep_mask       = S_i[neighbor_labels]

            if keep_mask.sum() < 2:
                competences[i] = knorau.DSEL_processed_[
                    neighbors, :].sum(axis=0).astype(np.float64)
                continue

            filtered     = neighbors[keep_mask]
            correct_mat  = knorau.DSEL_processed_[filtered, :]    # (f, n_clf)

            for c in range(n_clf):
                acc_local   = float(correct_mat[:, c].mean())
                probs_local = self._dsel_proba_cache[filtered, c, :]
                ece_local   = ece_per_neighbors(probs_local,
                                                correct_mat[:, c],
                                                n_bins=5)
                competences[i, c] = (1.0 - ece_local) * acc_local

        return competences


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _randomize_conformal_sets(S: np.ndarray) -> np.ndarray:
    """
    Replace each row of S with a random boolean set of matched cardinality.

    Preserves the marginal distribution of |S(x)| but uses random class
    membership.  Used for the CT-DES kill control experiment.

    Parameters
    ----------
    S : (n, K) bool
    Returns  : (n, K) bool — same row sums, random positions.
    """
    rng = np.random.default_rng(seed=42)
    n, K = S.shape
    S_rand = np.zeros_like(S)
    for i in range(n):
        card = int(S[i].sum())
        chosen = rng.choice(K, size=card, replace=False)
        S_rand[i, chosen] = True
    return S_rand
