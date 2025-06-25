"""
Feature normalization steps following the same pattern as graphic/steps/
"""

from typing import Optional, Dict, Any

import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler
)

from preprocessing.feature.base.step import BasePreprocessingStep


class NormalizationStep(BasePreprocessingStep):
    """Normalize features using various methods."""

    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scaler = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'NormalizationStep':
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        return self.scaler.transform(X)

    def get_params(self) -> Dict[str, Any]:
        params = {'method': self.method}
        if self.scaler and hasattr(self.scaler, 'mean_'):
            params['mean'] = self.scaler.mean_
            params['scale'] = self.scaler.scale_
        return params