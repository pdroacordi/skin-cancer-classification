"""
Feature selection steps following the same pattern as graphic/steps/
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif

from ..base.step import BasePreprocessingStep


class FeatureSelectionStep(BasePreprocessingStep):
    """Select the most relevant features from a CNN feature matrix.

    Four methods are supported:

    * ``mutual_info``   — SelectKBest with mutual information score.
      Non-parametric; handles non-linear dependencies.  Preferred default.
    * ``f_score``       — SelectKBest with ANOVA F-statistic.  Assumes linear
      separability; faster than mutual_info on large feature sets.
    * ``rfe``           — Recursive Feature Elimination backed by a lightweight
      RandomForest (50 trees).  The shallow forest is intentional: RFE calls
      ``fit`` ~1/step times, so a large estimator would be prohibitively slow.
      50 trees gives a stable importance ranking without the full training cost.
    * ``importance_based`` — Fit a full 100-tree RandomForest once, then keep the
      top-k features by Gini importance.  Unlike RFE it is a single fit, so the
      100-tree budget is justified for the more accurate importance estimate.
    """

    def __init__(
        self,
        method: str = 'mutual_info',
        k: Optional[int] = None,
        percentile: float = 75,
    ):
        """
        Args:
            method:     Selection algorithm (see class docstring).
            k:          Exact number of features to select.  If ``None``,
                        ``percentile`` is used to derive it from the input width.
            percentile: Fraction of features to keep (0–100).  Ignored when
                        ``k`` is given explicitly.
        """
        self.method = method
        self.k = k
        self.percentile = percentile
        self.selector = None
        self.selected_indices: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureSelectionStep':
        if y is None:
            raise ValueError("Feature selection requires labels")

        # Derive k from percentile on each fit call so the selector adapts if
        # the input width changes between runs (e.g. after a prior pipeline step
        # changes the feature count).  An explicitly-set self.k always wins.
        k = self.k if self.k is not None else int(X.shape[1] * self.percentile / 100)
        # Cache the resolved k so get_params() can report it.
        self._resolved_k = k

        if self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
            self.selector.fit(X, y)

        elif self.method == 'f_score':
            self.selector = SelectKBest(score_func=f_classif, k=k)
            self.selector.fit(X, y)

        elif self.method == 'rfe':
            # 50-tree forest: enough for a stable feature ranking, much faster
            # than a full forest because RFE refits it ~1/step iterations.
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            self.selector = RFE(estimator=estimator, n_features_to_select=k, step=0.1)
            self.selector.fit(X, y)

        elif self.method == 'importance_based':
            # Single full fit (100 trees) gives a more accurate Gini importance
            # estimate than the 50-tree RFE estimator.  We sort by importance and
            # keep the top-k indices so transform() is a simple column slice.
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            self.selected_indices = np.argsort(rf.feature_importances_)[::-1][:k]

        else:
            raise ValueError(
                f"Unknown feature selection method: '{self.method}'. "
                "Choose from 'mutual_info', 'f_score', 'rfe', 'importance_based'."
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == 'importance_based':
            if self.selected_indices is None:
                raise RuntimeError("Call fit() before transform()")
            return X[:, self.selected_indices]

        if self.selector is None:
            raise RuntimeError("Call fit() before transform()")
        return self.selector.transform(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'k_configured': self.k,
            'percentile': self.percentile,
            'n_features_selected': getattr(self, '_resolved_k', self.k),
        }
