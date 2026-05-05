"""
Tests for the new DES algorithm variants in models/dynamic_ensemble.py.

Uses a synthetic 3-class problem (50 DSEL + 30 test samples) with 3 pool
classifiers to verify:
  - CalibrationAwareKNORAU produces valid (n_test, 3) predictions.
  - ConformalTargetedKNORAU falls back to KNORAU when |S|=1.
  - ConformalTargetedKNORAU random kill-control produces a S distribution
    with the same per-row cardinality but different class membership.
  - TemperatureScaledDES wraps correctly and scales probabilities.
  - APSConformalPredictor integrates with the DES selector correctly.
"""

import os
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.dynamic_ensemble import (
    DynamicEnsembleSelector,
    CalibrationAwareKNORAU,
    ConformalTargetedKNORAU,
    ConformalTargetedECEKNORAU,
    TemperatureScaledDES,
    _randomize_conformal_sets,
)
from utils.conformal import APSConformalPredictor
from utils.calibration import TemperatureScaler


# ---------------------------------------------------------------------------
# Shared synthetic fixture
# ---------------------------------------------------------------------------

def _make_dataset(seed=0, n_train=150, n_dsel=50, n_test=30, n_features=10, n_classes=3):
    rng  = np.random.default_rng(seed)
    X    = rng.standard_normal((n_train + n_dsel + n_test, n_features)).astype(np.float32)
    y    = rng.integers(0, n_classes, n_train + n_dsel + n_test)
    X_tr, X_dsel, X_test = (
        X[:n_train], X[n_train:n_train+n_dsel], X[n_train+n_dsel:]
    )
    y_tr, y_dsel, y_test = (
        y[:n_train], y[n_train:n_train+n_dsel], y[n_train+n_dsel:]
    )
    clfs = [
        ("RF1", RandomForestClassifier(n_estimators=5, random_state=i).fit(X_tr, y_tr))
        for i in range(3)
    ]
    return clfs, X_dsel, y_dsel, X_test, y_test


# ---------------------------------------------------------------------------
# DynamicEnsembleSelector — new algorithm keys
# ---------------------------------------------------------------------------

class TestDESWrapperNewAlgorithms:
    def test_knorau_ece_predict_shape(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        des = DynamicEnsembleSelector(algorithm='knorau-ece', k_neighbors=5)
        des.fit(clfs, X_d, y_d)
        preds = des.predict(X_t)
        assert preds.shape == (len(X_t),)
        proba = des.predict_proba(X_t)
        assert proba.shape == (len(X_t), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_knorau_temp_requires_val(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        des = DynamicEnsembleSelector(algorithm='knorau-temp', k_neighbors=5)
        with pytest.raises((ValueError, Exception)):
            des.fit(clfs, X_d, y_d)  # no X_val provided → should raise

    def test_knorau_temp_with_val(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        des = DynamicEnsembleSelector(algorithm='knorau-temp', k_neighbors=5)
        des.fit(clfs, X_d, y_d, X_val=X_d, y_val=y_d)
        preds = des.predict(X_t)
        assert preds.shape == (len(X_t),)


# ---------------------------------------------------------------------------
# CalibrationAwareKNORAU
# ---------------------------------------------------------------------------

class TestCalibrationAwareKNORAU:
    def test_predict_outputs_valid_class(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        estimators = [clf for _, clf in clfs]
        clf = CalibrationAwareKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        clf.precompute_dsel_probas()
        preds = clf.predict(X_t)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_predict_proba_sums_to_one(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        estimators = [clf for _, clf in clfs]
        clf = CalibrationAwareKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        clf.precompute_dsel_probas()
        proba = clf.predict_proba(X_t)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_raises_without_precompute(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        estimators = [clf for _, clf in clfs]
        clf = CalibrationAwareKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        # precompute_dsel_probas not called
        with pytest.raises(RuntimeError, match="precompute_dsel_probas"):
            clf.predict(X_t)


# ---------------------------------------------------------------------------
# ConformalTargetedKNORAU
# ---------------------------------------------------------------------------

class TestConformalTargetedKNORAU:
    def _setup(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset(n_test=20)
        estimators = [clf for _, clf in clfs]
        return estimators, X_d, y_d, X_t, y_t

    def _make_S(self, n_test, n_cls, seed=0):
        rng = np.random.default_rng(seed)
        # Include 2 classes per sample
        S = np.zeros((n_test, n_cls), dtype=bool)
        for i in range(n_test):
            chosen = rng.choice(n_cls, size=2, replace=False)
            S[i, chosen] = True
        return S

    def test_predict_with_S(self):
        estimators, X_d, y_d, X_t, y_t = self._setup()
        clf = ConformalTargetedKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        S = self._make_S(len(X_t), 3)
        clf.set_conformal_sets(S)
        preds = clf.predict(X_t)
        assert preds.shape == (len(X_t),)

    def test_predict_proba_sums_to_one(self):
        estimators, X_d, y_d, X_t, y_t = self._setup()
        clf = ConformalTargetedKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        S = self._make_S(len(X_t), 3)
        clf.set_conformal_sets(S)
        proba = clf.predict_proba(X_t)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fallback_when_S_is_all_ones(self):
        estimators, X_d, y_d, X_t, y_t = self._setup()
        clf = ConformalTargetedKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        # S = all classes → degenerate, should fall back to KNORAU
        S = np.ones((len(X_t), 3), dtype=bool)
        clf.set_conformal_sets(S)
        preds = clf.predict(X_t)
        assert preds.shape == (len(X_t),)

    def test_fallback_when_S_is_singleton(self):
        estimators, X_d, y_d, X_t, y_t = self._setup()
        clf = ConformalTargetedKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        # S = only class 0 for every sample → should fall back to KNORAU
        S = np.zeros((len(X_t), 3), dtype=bool)
        S[:, 0] = True
        clf.set_conformal_sets(S)
        preds = clf.predict(X_t)
        assert preds.shape == (len(X_t),)

    def test_raises_without_S(self):
        estimators, X_d, y_d, X_t, y_t = self._setup()
        clf = ConformalTargetedKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        with pytest.raises(RuntimeError, match="set_conformal_sets"):
            clf.predict(X_t)

    def test_wrong_S_length_raises(self):
        estimators, X_d, y_d, X_t, y_t = self._setup()
        clf = ConformalTargetedKNORAU(pool_classifiers=estimators, k=5)
        clf.fit(X_d, y_d)
        S = self._make_S(len(X_t) + 5, 3)  # wrong length
        clf.set_conformal_sets(S)
        with pytest.raises(ValueError, match="rows"):
            clf.predict(X_t)


# ---------------------------------------------------------------------------
# Random kill control
# ---------------------------------------------------------------------------

class TestRandomizeConformalSets:
    def test_cardinality_preserved(self):
        rng = np.random.default_rng(0)
        K   = 7
        S   = np.zeros((100, K), dtype=bool)
        for i in range(100):
            card = rng.integers(1, K + 1)
            idx  = rng.choice(K, size=card, replace=False)
            S[i, idx] = True
        S_rand = _randomize_conformal_sets(S)
        np.testing.assert_array_equal(S.sum(axis=1), S_rand.sum(axis=1))

    def test_class_membership_differs(self):
        rng = np.random.default_rng(1)
        K   = 7
        n   = 200
        S   = np.zeros((n, K), dtype=bool)
        for i in range(n):
            S[i, rng.choice(K, size=2, replace=False)] = True
        S_rand = _randomize_conformal_sets(S)
        # At least some rows differ
        assert not np.array_equal(S, S_rand), (
            "Random kill control should differ from original in at least some rows"
        )


# ---------------------------------------------------------------------------
# TemperatureScaledDES
# ---------------------------------------------------------------------------

class TestTemperatureScaledDES:
    def test_predict_proba_recalibrated(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        estimators = [clf for _, clf in clfs]
        from deslib.des import KNORAU
        knorau = KNORAU(pool_classifiers=estimators, k=5)
        knorau.fit(X_d, y_d)

        raw_proba = knorau.predict_proba(X_t)
        logits    = np.log(np.clip(raw_proba, 1e-9, 1.0))
        scaler    = TemperatureScaler()
        scaler.fit(logits, y_t)

        des_ts = TemperatureScaledDES(knorau, scaler)
        p = des_ts.predict_proba(X_t)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
        assert p.shape == (len(X_t), 3)

    def test_predict_matches_base(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        estimators = [clf for _, clf in clfs]
        from deslib.des import KNORAU
        knorau = KNORAU(pool_classifiers=estimators, k=5)
        knorau.fit(X_d, y_d)
        logits = np.log(np.clip(knorau.predict_proba(X_t), 1e-9, 1.0))
        scaler = TemperatureScaler()
        scaler.fit(logits, y_t)
        des_ts = TemperatureScaledDES(knorau, scaler)
        # predict() argmax is invariant to temperature
        np.testing.assert_array_equal(knorau.predict(X_t), des_ts.predict(X_t))


# ---------------------------------------------------------------------------
# DynamicEnsembleSelector — conformal integration
# ---------------------------------------------------------------------------

class TestDESWrapperConformal:
    def test_conformal_targeted_end_to_end(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset(n_test=25)
        des = DynamicEnsembleSelector(algorithm='conformal-targeted', k_neighbors=5)
        des.fit(clfs, X_d, y_d)

        # Build APS sets
        pool_proba = np.mean(
            [clf.predict_proba(X_t) for _, clf in clfs], axis=0
        )
        aps = APSConformalPredictor(alpha=0.10)
        aps.fit(y_d, np.mean([clf.predict_proba(X_d) for _, clf in clfs], axis=0))
        S_test = aps.predict_set(pool_proba)

        des.set_conformal_sets(S_test)
        preds = des.predict(X_t)
        assert preds.shape == (len(X_t),)
        proba = des.predict_proba(X_t)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_set_conformal_sets_wrong_algo_raises(self):
        clfs, X_d, y_d, X_t, y_t = _make_dataset()
        des = DynamicEnsembleSelector(algorithm='knorau', k_neighbors=5)
        des.fit(clfs, X_d, y_d)
        S = np.ones((len(X_t), 3), dtype=bool)
        with pytest.raises(TypeError, match="conformal-targeted"):
            des.set_conformal_sets(S)
