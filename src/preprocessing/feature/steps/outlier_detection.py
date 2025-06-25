from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from ..base.preprocessor import (
    UnsupervisedFeaturePreprocessor, FeatureTransformationMixin
)


class OutlierDetector(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Detect and handle outliers in features."""

    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.05, **kwargs):
        """
        Initialize outlier detector.

        Args:
            method: Detection method ('isolation_forest', 'local_outlier_factor', 'one_class_svm')
            contamination: Expected proportion of outliers
            **kwargs: Additional arguments for the detector
        """
        self.method = method
        self.contamination = contamination
        self.detector_kwargs = kwargs
        self.detector = None
        self.outlier_mask = None

    def _fit(self, features: np.ndarray) -> 'OutlierDetector':
        """Fit the outlier detector."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        if self.method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
                **self.detector_kwargs
            )

        elif self.method == 'local_outlier_factor':
            self.detector = LocalOutlierFactor(
                contamination=self.contamination,
                n_jobs=-1,
                **self.detector_kwargs
            )

        elif self.method == 'one_class_svm':
            self.detector = OneClassSVM(
                nu=self.contamination,
                **self.detector_kwargs
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit detector and get outlier mask
        if self.method == 'local_outlier_factor':
            # LOF doesn't have separate fit/predict
            outlier_labels = self.detector.fit_predict(features)
        else:
            self.detector.fit(features)
            outlier_labels = self.detector.predict(features)

        # Convert to boolean mask (True = inlier, False = outlier)
        self.outlier_mask = outlier_labels == 1

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform by removing outliers (for training) or returning as-is (for prediction)."""
        self._validate_features(features)

        # For transform, we don't actually remove outliers, just return features
        # Outlier removal should only happen during training
        return self._ensure_float32(features)

    def get_outlier_mask(self) -> Optional[np.ndarray]:
        """Get the outlier mask from training."""
        return self.outlier_mask

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Outlier Detection ({self.method})"

    def get_params(self) -> dict:
        """Get detector parameters."""
        params = {
            'method': self.method,
            'contamination': self.contamination,
            'n_outliers_detected': np.sum(~self.outlier_mask) if self.outlier_mask is not None else None
        }
        if self.detector:
            params.update(self.detector.get_params())
        return params

