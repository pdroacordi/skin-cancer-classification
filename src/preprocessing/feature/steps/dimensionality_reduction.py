from typing import Optional

import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from ..base.preprocessor import (
    UnsupervisedFeaturePreprocessor, FeatureTransformationMixin
)


class PCAReducer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """PCA-based dimensionality reduction."""

    def __init__(self, variance_threshold: float = 0.95,
                 max_components: Optional[int] = None,
                 whiten: bool = False):
        """
        Initialize PCA reducer.

        Args:
            variance_threshold: Cumulative variance to retain
            max_components: Maximum number of components
            whiten: Whether to whiten the components
        """
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.whiten = whiten
        self.pca = None
        self.n_components = None

    def _fit(self, features: np.ndarray) -> 'PCAReducer':
        """Fit PCA."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        # Determine number of components
        if self.max_components:
            n_components = min(self.max_components, features.shape[1], features.shape[0])
        else:
            n_components = min(features.shape[1], features.shape[0])

        # Fit PCA
        self.pca = PCA(n_components=n_components, whiten=self.whiten, random_state=42)
        self.pca.fit(features)

        # Determine final number of components based on variance threshold
        cumsum_var = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1

        # Refit with final number of components
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten, random_state=42)
        self.pca.fit(features)

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform using PCA."""
        self._validate_features(features)

        if self.pca is None:
            return self._ensure_float32(features)

        transformed = self.pca.transform(features)
        return self._ensure_float32(transformed)

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"PCA Reduction ({self.n_components} components)"

    def get_params(self) -> dict:
        """Get PCA parameters."""
        params = {
            'variance_threshold': self.variance_threshold,
            'max_components': self.max_components,
            'whiten': self.whiten,
            'n_components': self.n_components
        }

        if self.pca:
            params.update({
                'explained_variance_ratio': self.pca.explained_variance_ratio_[:5].tolist(),  # First 5
                'total_explained_variance': np.sum(self.pca.explained_variance_ratio_)
            })

        return params


class KernelPCAReducer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Kernel PCA-based non-linear dimensionality reduction."""

    def __init__(self, n_components: Optional[int] = None,
                 kernel: str = 'rbf', gamma: Optional[float] = None):
        """
        Initialize Kernel PCA reducer.

        Args:
            n_components: Number of components to keep
            kernel: Kernel type ('rbf', 'poly', 'sigmoid', 'cosine')
            gamma: Kernel coefficient
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.kpca = None

    def _fit(self, features: np.ndarray) -> 'KernelPCAReducer':
        """Fit Kernel PCA."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        # Determine number of components
        if self.n_components is None:
            self.n_components = min(100, features.shape[1] // 2, features.shape[0] // 2)

        # Determine gamma
        gamma = self.gamma
        if gamma is None and self.kernel == 'rbf':
            gamma = 1.0 / features.shape[1]

        # Fit Kernel PCA
        self.kpca = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=gamma,
            random_state=42,
            n_jobs=-1
        )

        self.kpca.fit(features)

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform using Kernel PCA."""
        self._validate_features(features)

        if self.kpca is None:
            return self._ensure_float32(features)

        try:
            transformed = self.kpca.transform(features)
            return self._ensure_float32(transformed)
        except Exception as e:
            print(f"Warning: Kernel PCA transform failed: {e}")
            return self._ensure_float32(features)

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Kernel PCA Reduction ({self.kernel}, {self.n_components} components)"

    def get_params(self) -> dict:
        """Get Kernel PCA parameters."""
        return {
            'n_components': self.n_components,
            'kernel': self.kernel,
            'gamma': self.gamma
        }