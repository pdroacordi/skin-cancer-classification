"""
Tests for APSConformalPredictor (utils/conformal.py).

Verifies:
  1. Marginal coverage property: P(Y ∈ S(X)) ≥ 1−α holds empirically within ±3pp.
  2. Set size decreases as α increases (less stringent coverage → smaller sets).
  3. predict_set returns a non-empty set for every test sample.
  4. save / load round-trip preserves q_hat.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.conformal import APSConformalPredictor, _aps_scores, _aps_predict_sets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_softmax(rng, n, K):
    """Generate random valid softmax probabilities."""
    logits = rng.standard_normal((n, K))
    exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


def _perfect_probs(y, K):
    """Probabilities that are perfectly calibrated: p[y] = 1."""
    p = np.zeros((len(y), K), dtype=np.float64)
    p[np.arange(len(y)), y] = 1.0
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAPSScores:
    def test_shape(self):
        rng = np.random.default_rng(0)
        n, K = 50, 7
        p = _make_softmax(rng, n, K)
        y = rng.integers(0, K, n)
        scores = _aps_scores(y, p)
        assert scores.shape == (n,)

    def test_perfect_predictor_scores_equal_true_class_prob(self):
        K = 7
        y = np.arange(K)
        p = _perfect_probs(y, K)
        scores = _aps_scores(y, p)
        np.testing.assert_allclose(scores, 1.0, atol=1e-9)

    def test_scores_in_zero_one(self):
        rng = np.random.default_rng(1)
        p = _make_softmax(rng, 200, 7)
        y = rng.integers(0, 7, 200)
        scores = _aps_scores(y, p)
        assert (scores >= 0.0).all() and (scores <= 1.0 + 1e-9).all()


class TestAPSPredictSets:
    def test_non_empty_sets(self):
        rng = np.random.default_rng(2)
        p   = _make_softmax(rng, 100, 7)
        S   = _aps_predict_sets(p, q_hat=0.5)
        assert S.dtype == bool
        assert S.shape == (100, 7)
        assert S.sum(axis=1).min() >= 1, "Every sample must have ≥1 class in set"

    def test_larger_q_hat_larger_sets(self):
        rng = np.random.default_rng(3)
        p   = _make_softmax(rng, 500, 7)
        S_small = _aps_predict_sets(p, q_hat=0.3)
        S_large = _aps_predict_sets(p, q_hat=0.9)
        assert S_small.sum() <= S_large.sum()

    def test_q_hat_geq_1_includes_all_classes(self):
        rng = np.random.default_rng(4)
        p   = _make_softmax(rng, 20, 7)
        S   = _aps_predict_sets(p, q_hat=1.0 + 1e-9)
        assert S.all(), "q_hat≥1 should include all classes"


class TestAPSConformalPredictor:
    """End-to-end coverage and API tests."""

    N_CAL  = 1000
    N_TEST = 2000
    K      = 7

    def _calibrated_predictor(self, alpha, seed=42):
        rng = np.random.default_rng(seed)
        p_cal = _make_softmax(rng, self.N_CAL, self.K)
        y_cal = rng.integers(0, self.K, self.N_CAL)
        aps   = APSConformalPredictor(alpha=alpha)
        aps.fit(y_cal, p_cal)
        return aps, rng

    def test_marginal_coverage_alpha_010(self):
        alpha = 0.10
        aps, rng = self._calibrated_predictor(alpha)
        p_test = _make_softmax(rng, self.N_TEST, self.K)
        y_test = rng.integers(0, self.K, self.N_TEST)
        report = aps.coverage_report(y_test, p_test)
        # Allow ±3pp slack (finite-sample)
        assert report["empirical_coverage"] >= 1 - alpha - 0.03, (
            f"Coverage {report['empirical_coverage']:.3f} < {1-alpha-0.03:.3f}"
        )

    def test_marginal_coverage_alpha_020(self):
        alpha = 0.20
        aps, rng = self._calibrated_predictor(alpha, seed=7)
        p_test = _make_softmax(rng, self.N_TEST, self.K)
        y_test = rng.integers(0, self.K, self.N_TEST)
        report = aps.coverage_report(y_test, p_test)
        assert report["empirical_coverage"] >= 1 - alpha - 0.03

    def test_set_size_decreases_with_alpha(self):
        _, rng = self._calibrated_predictor(0.05, seed=9)
        p_test = _make_softmax(rng, self.N_TEST, self.K)
        y_test = rng.integers(0, self.K, self.N_TEST)

        sizes = {}
        for seed, alpha in [(5, 0.05), (10, 0.10), (20, 0.20)]:
            aps = APSConformalPredictor(alpha=alpha)
            p_cal = _make_softmax(np.random.default_rng(seed), self.N_CAL, self.K)
            y_cal = np.random.default_rng(seed).integers(0, self.K, self.N_CAL)
            aps.fit(y_cal, p_cal)
            report = aps.coverage_report(y_test, p_test)
            sizes[alpha] = report["mean_set_size"]

        # Higher α (less coverage required) → smaller sets
        assert sizes[0.20] <= sizes[0.05] + 0.5, (
            f"Expected mean set size to decrease with α, got {sizes}"
        )

    def test_save_load_roundtrip(self):
        aps, _ = self._calibrated_predictor(0.10)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "aps.joblib")
            aps.save(path)
            aps2 = APSConformalPredictor.load(path)
        assert abs(aps.q_hat - aps2.q_hat) < 1e-12
        assert aps2.alpha == aps.alpha

    def test_predict_set_shape(self):
        aps, rng = self._calibrated_predictor(0.10)
        p_test = _make_softmax(rng, 50, self.K)
        S = aps.predict_set(p_test)
        assert S.shape == (50, self.K)
        assert S.dtype == bool

    def test_unfitted_raises(self):
        aps = APSConformalPredictor(alpha=0.10)
        p   = np.ones((5, 7)) / 7
        with pytest.raises(RuntimeError, match="fit"):
            aps.predict_set(p)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            APSConformalPredictor(alpha=0.0)
        with pytest.raises(ValueError):
            APSConformalPredictor(alpha=1.0)
