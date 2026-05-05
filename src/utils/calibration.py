"""
Model calibration utilities.

Provides Expected Calibration Error (ECE) computation for evaluating
how well a model's confidence scores match observed frequencies.

ECE reference: Guo et al. (2017, ICML, "On Calibration of Modern Neural Networks").
Medical imaging context: Nixon et al. (2019, CVPR Workshop).
"""

import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax


def ece_per_neighbors(probs: np.ndarray, correct: np.ndarray, n_bins: int = 5) -> float:
    """
    ECE computed on a small local neighborhood (K~7 neighbors).

    Uses fewer bins than the global ECE (5 vs 15) to avoid empty bins when
    sample count equals the DES neighborhood size K.

    Parameters
    ----------
    probs   : (K, n_classes) float — per-neighbor predicted probabilities from
              ONE classifier.  Uses max confidence as the confidence value.
    correct : (K,) bool/int — per-neighbor correctness indicator.
    n_bins  : int — number of equal-width bins (default 5).

    Returns
    -------
    float — ECE in [0, 1].  Returns 0.0 when K < 2 (trivially calibrated).
    """
    if len(probs) < 2:
        return 0.0

    confidences = np.max(probs, axis=1)          # (K,)
    correct_f   = np.asarray(correct, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n_in = mask.sum()
        if n_in > 0:
            ece += n_in * abs(correct_f[mask].mean() - confidences[mask].mean())

    return ece / max(len(probs), 1)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the gap between a model's confidence and its actual accuracy,
    averaged across confidence bins. A well-calibrated model predicting 80%
    confidence should be correct 80% of the time.

    Args:
        y_true: 1-D integer array of true class indices, shape (n_samples,).
        y_prob: 2-D float array of softmax probabilities, shape (n_samples, n_classes).
        n_bins: Number of equally-spaced confidence bins (default: 15).

    Returns:
        float: ECE in [0, 1]. Lower is better. 0 = perfectly calibrated.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += n_in_bin * abs(bin_accuracy - bin_confidence)

    return ece / len(y_true)


class TemperatureScaler:
    """
    Post-hoc temperature scaling (Guo et al., 2017).

    Fits a single scalar T by minimising NLL on a held-out validation set
    (typically the fold's val split). At inference time, softmax(logits / T)
    produces better-calibrated probabilities without altering the argmax.
    """

    def __init__(self, T: float = 1.0):
        self.T = float(T)

    def fit(self, logits_val: np.ndarray, y_val: np.ndarray):
        """
        Fit T on validation logits via L-BFGS on NLL.

        Args:
            logits_val: (n_samples, n_classes) pre-softmax activations.
            y_val: (n_samples,) integer class indices.

        Returns:
            self.
        """
        logits_val = np.asarray(logits_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.int64)

        def nll(t):
            T = max(float(t[0]), 1e-3)
            log_p = log_softmax(logits_val / T, axis=1)
            return -np.mean(log_p[np.arange(len(y_val)), y_val])

        res = minimize(
            nll, x0=np.array([1.0]), method='L-BFGS-B',
            bounds=[(0.05, 10.0)],
        )
        self.T = float(np.clip(res.x[0], 0.05, 10.0))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Return softmax(logits / T) as probabilities."""
        logits = np.asarray(logits, dtype=np.float64)
        # Numerically stable softmax of logits / T.
        scaled = logits / self.T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({'T': self.T}, f)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        with open(path) as f:
            data = json.load(f)
        return cls(T=float(data['T']))
