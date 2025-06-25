from typing import Optional, Dict, Any

import numpy as np

from preprocessing.feature.base.step import BasePreprocessingStep


class VarianceThresholdStep(BasePreprocessingStep):
    """Remove low variance features."""

    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.selector = None
        self.n_features_removed = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'VarianceThresholdStep':
        from sklearn.feature_selection import VarianceThreshold
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(X)
        self.n_features_removed = X.shape[1] - sum(self.selector.get_support())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selector is None:
            return X
        return self.selector.transform(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            'threshold': self.threshold,
            'n_features_removed': self.n_features_removed
        }