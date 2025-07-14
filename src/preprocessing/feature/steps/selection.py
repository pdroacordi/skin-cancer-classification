"""
Feature selection steps following the same pattern as graphic/steps/
"""

from typing import Optional, Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, RFE, f_classif, mutual_info_classif
)

from ..base.step import BasePreprocessingStep


class FeatureSelectionStep(BasePreprocessingStep):
    """Select most relevant features."""

    def __init__(self, method: str = 'mutual_info', k: Optional[int] = None,
                 percentile: float = 75):
        self.method = method
        self.k = k
        self.percentile = percentile
        self.selector = None
        self.selected_indices = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureSelectionStep':
        if y is None:
            raise ValueError("Feature selection requires labels")

        # Calculate k from percentile if not provided
        if self.k is None:
            self.k = int(X.shape[1] * self.percentile / 100)

        if self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
        elif self.method == 'f_score':
            self.selector = SelectKBest(score_func=f_classif, k=self.k)
        elif self.method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            self.selector = RFE(estimator=estimator, n_features_to_select=self.k, step=0.1)
        elif self.method == 'importance_based':
            # Custom importance-based selection
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            self.selected_indices = np.argsort(importances)[::-1][:self.k]
            return self

        self.selector.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == 'importance_based' and self.selected_indices is not None:
            return X[:, self.selected_indices]
        elif self.selector is not None:
            return self.selector.transform(X)
        return X

    def get_params(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'k': self.k,
            'percentile': self.percentile,
            'n_features_selected': self.k
        }


class CorrelationBasedSelection(BasePreprocessingStep):
    """Remove highly correlated features to reduce redundancy."""

    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.features_to_keep = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CorrelationBasedSelection':
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)

        # Find highly correlated feature pairs
        upper_triangle = np.triu(np.abs(corr_matrix), k=1)

        # Features to remove
        features_to_remove = set()

        for i in range(len(upper_triangle)):
            for j in range(i + 1, len(upper_triangle)):
                if upper_triangle[i, j] > self.correlation_threshold:
                    # Remove the feature with lower variance
                    var_i = np.var(X[:, i])
                    var_j = np.var(X[:, j])
                    if var_i < var_j:
                        features_to_remove.add(i)
                    else:
                        features_to_remove.add(j)

        # Features to keep
        all_features = set(range(X.shape[1]))
        self.features_to_keep = sorted(list(all_features - features_to_remove))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.features_to_keep is None:
            return X
        return X[:, self.features_to_keep]

    def get_params(self) -> Dict[str, Any]:
        return {
            'correlation_threshold': self.correlation_threshold
        }