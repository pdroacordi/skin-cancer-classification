"""
Base classes for feature preprocessing following the same abstraction pattern
as the graphic preprocessing module.

This module provides the foundation for feature-level preprocessing steps
that can be composed into a pipeline, similar to graphic/base/preprocessor.py
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple


class FeaturePreprocessor(ABC):
    """Abstract base class for feature preprocessing steps."""

    @abstractmethod
    def process(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process the features and return the result.

        Args:
            features: Input feature matrix of shape (n_samples, n_features)
            labels: Optional target labels for supervised preprocessing

        Returns:
            Processed features of shape (n_samples, n_processed_features)
        """
        pass

    @abstractmethod
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> 'FeaturePreprocessor':
        """
        Fit the preprocessor to the training data.

        Args:
            features: Training feature matrix
            labels: Optional training labels

        Returns:
            self: Fitted preprocessor
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the preprocessor."""
        return {}

    def set_params(self, **params) -> 'FeaturePreprocessor':
        """Set parameters of the preprocessor."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the feature transformation."""
        return {
            'name': self.get_name(),
            'params': self.get_params(),
            'fitted': hasattr(self, '_is_fitted') and self._is_fitted
        }


class SupervisedFeaturePreprocessor(FeaturePreprocessor):
    """Base class for supervised feature preprocessing that requires labels."""

    def process(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Process features using fitted parameters."""
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError(f"{self.get_name()} must be fitted before processing")

        return self._transform(features)

    @abstractmethod
    def _fit(self, features: np.ndarray, labels: np.ndarray) -> 'SupervisedFeaturePreprocessor':
        """Internal fit method to be implemented by subclasses."""
        pass

    @abstractmethod
    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Internal transform method to be implemented by subclasses."""
        pass

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> 'SupervisedFeaturePreprocessor':
        """Fit the supervised preprocessor."""
        if labels is None:
            raise ValueError(f"{self.get_name()} requires labels for fitting")

        self._fit(features, labels)
        self._is_fitted = True
        return self


class UnsupervisedFeaturePreprocessor(FeaturePreprocessor):
    """Base class for unsupervised feature preprocessing that doesn't require labels."""

    def process(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Process features using fitted parameters."""
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError(f"{self.get_name()} must be fitted before processing")

        return self._transform(features)

    @abstractmethod
    def _fit(self, features: np.ndarray) -> 'UnsupervisedFeaturePreprocessor':
        """Internal fit method to be implemented by subclasses."""
        pass

    @abstractmethod
    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Internal transform method to be implemented by subclasses."""
        pass

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> 'UnsupervisedFeaturePreprocessor':
        """Fit the unsupervised preprocessor."""
        self._fit(features)
        self._is_fitted = True
        return self


class FeatureTransformationMixin:
    """Mixin class providing common feature transformation utilities."""

    @staticmethod
    def _ensure_float32(features: np.ndarray) -> np.ndarray:
        """Ensure features are float32 for memory efficiency."""
        return features.astype(np.float32)

    @staticmethod
    def _validate_features(features: np.ndarray) -> None:
        """Validate feature matrix format."""
        if not isinstance(features, np.ndarray):
            raise TypeError("Features must be a numpy array")

        if features.ndim != 2:
            raise ValueError(f"Features must be 2D, got {features.ndim}D")

        if features.size == 0:
            raise ValueError("Features array is empty")

        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or infinite values")

    @staticmethod
    def _validate_labels(labels: np.ndarray, n_samples: int) -> None:
        """Validate labels format."""
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a numpy array")

        if labels.ndim != 1:
            raise ValueError(f"Labels must be 1D, got {labels.ndim}D")

        if len(labels) != n_samples:
            raise ValueError(f"Labels length ({len(labels)}) doesn't match features ({n_samples})")

    @staticmethod
    def _get_transformation_info(self, input_shape: Tuple[int, int],
                                 output_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Get information about the transformation."""
        return {
            'input_features': input_shape[1],
            'output_features': output_shape[1],
            'n_samples': input_shape[0],
            'compression_ratio': output_shape[1] / input_shape[1] if input_shape[1] > 0 else 0,
            'transformation_type': 'compression' if output_shape[1] < input_shape[1] else 'expansion'
        }