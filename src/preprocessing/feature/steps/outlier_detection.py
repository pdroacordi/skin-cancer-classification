from typing import Optional, Dict, Any

import numpy as np
from sklearn.ensemble import IsolationForest

from preprocessing.feature.base.step import BasePreprocessingStep


class OutlierRemovalStep(BasePreprocessingStep):
    """Handle outliers in the data."""

    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.05):
        self.method = method
        self.contamination = contamination
        self.outlier_detector = None
        self.outlier_mask = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OutlierRemovalStep':
        if self.method == 'isolation_forest':
            self.outlier_detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            outlier_labels = self.outlier_detector.fit_predict(X)
            self.outlier_mask = outlier_labels == 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # For transform, we don't remove outliers (only mark them)
        # Actual removal happens in the pipeline during training
        return X

    def get_params(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'contamination': self.contamination,
            'n_outliers': np.sum(~self.outlier_mask) if self.outlier_mask is not None else 0
        }