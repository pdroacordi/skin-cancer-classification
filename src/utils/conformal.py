"""
Adaptive Prediction Sets (APS) conformal predictor.

Hand-rolled implementation of Romano, Sesia & Candès (2020), "Classification
with Valid and Adaptive Coverage," NeurIPS 2020.  No mapie dependency — APS is
~40 lines and the project already carries enough heavy packages.

APS nonconformity score for sample i with true class y:
    s(x, y) = Σ_{j: rank(j) ≤ rank(y)} p_j
             (cumulative probability from the most likely class down to y,
              in descending-probability order)

Calibration: q̂ = ⌈(n+1)(1-α)⌉/n quantile of scores on the calibration set.
Prediction set: S(x) = {y : s(x, y) ≤ q̂}

Coverage guarantee: P(Y ∈ S(X)) ≥ 1 - α (marginal).

Usage::
    from utils.conformal import APSConformalPredictor
    aps = APSConformalPredictor(alpha=0.10)
    aps.fit(y_val, p_val)          # calibrate on validation fold
    S = aps.predict_set(p_test)    # (n_test, n_classes) boolean matrix
    aps.save('path/conformal.joblib')
    aps2 = APSConformalPredictor.load('path/conformal.joblib')
"""

from __future__ import annotations

import joblib
import numpy as np


class APSConformalPredictor:
    """
    Adaptive Prediction Sets conformal predictor.

    Parameters
    ----------
    alpha : float
        Miscoverage level. Default 0.10 → prediction sets cover the true
        class with probability ≥ 0.90.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_hat: float | None = None
        self.n_classes_: int | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(self, y_cal: np.ndarray, p_cal: np.ndarray) -> "APSConformalPredictor":
        """
        Calibrate on a held-out validation set.

        Parameters
        ----------
        y_cal : (n_cal,) int array
            True class indices.
        p_cal : (n_cal, n_classes) float array
            Softmax probabilities from the model.

        Returns
        -------
        self
        """
        y_cal = np.asarray(y_cal, dtype=np.intp)
        p_cal = np.asarray(p_cal, dtype=np.float64)

        n, K = p_cal.shape
        self.n_classes_ = K

        scores = _aps_scores(y_cal, p_cal)

        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level))
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_set(self, p_test: np.ndarray) -> np.ndarray:
        """
        Return boolean prediction sets for test samples.

        Parameters
        ----------
        p_test : (n_test, n_classes) float array
            Softmax probabilities.

        Returns
        -------
        S : (n_test, n_classes) bool array
            S[i, c] = True iff class c is in the conformal prediction set
            for test sample i.
        """
        if self.q_hat is None:
            raise RuntimeError("Call fit() before predict_set().")

        p_test = np.asarray(p_test, dtype=np.float64)
        return _aps_predict_sets(p_test, self.q_hat)

    def coverage_report(self, y_test: np.ndarray, p_test: np.ndarray) -> dict:
        """
        Compute empirical coverage and mean set size on a test set.

        Returns a dict with keys: alpha, q_hat, empirical_coverage, mean_set_size.
        Empirical coverage should be ≥ 1−alpha (up to statistical fluctuation).
        """
        S = self.predict_set(p_test)
        y_test = np.asarray(y_test, dtype=np.intp)
        covered = S[np.arange(len(y_test)), y_test].mean()
        return {
            "alpha":              self.alpha,
            "q_hat":              self.q_hat,
            "empirical_coverage": float(covered),
            "mean_set_size":      float(S.sum(axis=1).mean()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"alpha": self.alpha, "q_hat": self.q_hat,
                     "n_classes_": self.n_classes_}, path)

    @classmethod
    def load(cls, path: str) -> "APSConformalPredictor":
        data = joblib.load(path)
        obj = cls(alpha=data["alpha"])
        obj.q_hat = data["q_hat"]
        obj.n_classes_ = data["n_classes_"]
        return obj


# ---------------------------------------------------------------------------
# Internal vectorized helpers
# ---------------------------------------------------------------------------

def _aps_scores(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Compute APS nonconformity scores for n samples.

    s(x_i, y_i) = sum of probabilities from the most likely class down to
    (and including) the true class y_i in descending-probability order.

    Parameters
    ----------
    y : (n,) int — true class indices.
    p : (n, K) float — softmax probabilities.

    Returns
    -------
    scores : (n,) float
    """
    n, K = p.shape
    # Sort class indices by descending probability for each sample
    sorted_idx = np.argsort(-p, axis=1)                         # (n, K)
    sorted_probs = p[np.arange(n)[:, None], sorted_idx]         # (n, K)
    cum_probs = np.cumsum(sorted_probs, axis=1)                  # (n, K)
    # rank_matrix[i, j] = position of class j in the sorted order for sample i
    rank_matrix = np.argsort(sorted_idx, axis=1)                 # (n, K)
    ranks = rank_matrix[np.arange(n), y]                        # (n,)
    return cum_probs[np.arange(n), ranks]                       # (n,)


def _aps_predict_sets(p: np.ndarray, q_hat: float) -> np.ndarray:
    """
    Build APS prediction sets for n_test samples.

    Include classes in descending-probability order until the cumulative
    probability exceeds q_hat (include the class that caused the crossing).

    Parameters
    ----------
    p      : (n_test, K) float — softmax probabilities.
    q_hat  : float — calibrated quantile threshold.

    Returns
    -------
    S : (n_test, K) bool
    """
    n, K = p.shape
    sorted_idx = np.argsort(-p, axis=1)                         # (n, K)
    sorted_probs = p[np.arange(n)[:, None], sorted_idx]         # (n, K)
    cum_probs = np.cumsum(sorted_probs, axis=1)                  # (n, K)

    # For each sample, include ranks 0..first_exceed (inclusive).
    # np.argmax returns 0 when no element is True (handled below).
    exceeds = cum_probs > q_hat                                  # (n, K)
    any_exceeds = exceeds.any(axis=1)                            # (n,)

    # first_exceed[i] = index of first rank whose cumulative prob > q_hat
    first_exceed = np.where(any_exceeds, np.argmax(exceeds, axis=1), K - 1)  # (n,)

    rank_mask = np.arange(K)[None, :] <= first_exceed[:, None]  # (n, K) bool
    # When no class exceeds q_hat, include all classes (ensures non-empty set)
    rank_mask[~any_exceeds] = True

    # Map rank-space mask back to class-space boolean matrix
    S = np.zeros((n, K), dtype=bool)
    np.put_along_axis(S, sorted_idx, rank_mask, axis=1)
    return S
