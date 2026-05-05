"""
Tests for ece_per_neighbors (utils/calibration.py).

Verifies shape, edge cases, and expected behaviour on constructed inputs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.calibration import ece_per_neighbors


class TestECEPerNeighbors:
    K     = 7
    N_CLS = 7

    def _one_hot_probs(self, cls, K):
        p = np.zeros((1, K), dtype=np.float32)
        p[0, cls] = 1.0
        return p

    def test_returns_float(self):
        rng  = np.random.default_rng(0)
        n    = self.K
        probs = rng.dirichlet(np.ones(self.N_CLS), size=n).astype(np.float32)
        correct = (np.argmax(probs, axis=1) == rng.integers(0, self.N_CLS, n))
        result  = ece_per_neighbors(probs, correct)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_returns_zero_for_K_less_than_2(self):
        # Edge case: single neighbor — trivially calibrated
        p = np.array([[0.8, 0.1, 0.1]], dtype=np.float32)
        c = np.array([1], dtype=bool)
        assert ece_per_neighbors(p, c) == 0.0

    def test_returns_zero_for_empty(self):
        p = np.zeros((0, 7), dtype=np.float32)
        c = np.zeros(0, dtype=bool)
        assert ece_per_neighbors(p, c) == 0.0

    def test_perfectly_calibrated_is_near_zero(self):
        # Confidence 1.0 and always correct → ECE ≈ 0
        K = self.K
        p = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            p[i, i % K] = 1.0
        correct = np.ones(K, dtype=bool)
        ece = ece_per_neighbors(p, correct, n_bins=5)
        assert ece < 0.05

    def test_maximally_miscalibrated(self):
        # Confidence 1.0 but always wrong → high ECE
        K = self.K
        p = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            p[i, (i + 1) % K] = 1.0   # argmax is never the right class
        correct = np.zeros(K, dtype=bool)
        ece = ece_per_neighbors(p, correct, n_bins=5)
        # |confidence - accuracy| = |1 - 0| = 1 in every bin
        assert ece > 0.5

    def test_n_bins_parameter_accepted(self):
        rng    = np.random.default_rng(1)
        probs  = rng.dirichlet(np.ones(7), size=7).astype(np.float32)
        correct = (np.argmax(probs, axis=1) == rng.integers(0, 7, 7))
        # Should not raise for any reasonable n_bins
        for nb in [1, 3, 5, 10]:
            ece_per_neighbors(probs, correct, n_bins=nb)
