from typing import Optional, Any, Dict

import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from preprocessing.feature.base.step import BasePreprocessingStep


class DimensionalityReductionStep(BasePreprocessingStep):
    """Reduce dimensionality using PCA."""

    def __init__(self, method: str = 'pca', variance_threshold: float = 0.95):
        self.method = method
        self.variance_threshold = variance_threshold
        self.reducer = None
        self.n_components = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DimensionalityReductionStep':
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.variance_threshold, random_state=42)
            self.reducer.fit(X)
            self.n_components = self.reducer.n_components_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            return X
        return self.reducer.transform(X)

    def get_params(self) -> Dict[str, Any]:
        params = {
            'method': self.method,
            'variance_threshold': self.variance_threshold,
            'n_components': self.n_components
        }
        if self.reducer and hasattr(self.reducer, 'explained_variance_ratio_'):
            params['explained_variance'] = np.sum(self.reducer.explained_variance_ratio_)
        return params