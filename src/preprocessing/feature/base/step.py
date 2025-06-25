"""
Base classes for feature preprocessing following the same abstraction pattern
as the graphic preprocessing module.

This module provides the foundation for feature-level preprocessing steps
that can be composed into a pipeline, similar to graphic/base/preprocessor.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np


class BasePreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BasePreprocessingStep':
        """Fit the preprocessing step."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the step."""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)