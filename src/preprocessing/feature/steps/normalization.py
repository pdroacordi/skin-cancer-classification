"""
Feature normalization steps following the same pattern as graphic/steps/
"""

from typing import Optional

import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer,
    QuantileTransformer, MinMaxScaler
)

from ..base.preprocessor import UnsupervisedFeaturePreprocessor, FeatureTransformationMixin


class FeatureNormalizer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Feature normalization using various scaling methods."""

    def __init__(self, method: str = 'robust', **kwargs):
        """
        Initialize feature normalizer.

        Args:
            method: Normalization method ('standard', 'robust', 'power', 'quantile', 'minmax', 'none')
            **kwargs: Additional arguments for the specific scaler
        """
        self.method = method
        self.scaler_kwargs = kwargs
        self.scaler = None

        valid_methods = ['standard', 'robust', 'power', 'quantile', 'minmax', 'none']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def _fit(self, features: np.ndarray) -> 'FeatureNormalizer':
        """Fit the normalizer to features."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        if self.method == 'none':
            self.scaler = None
            return self

        # Create the appropriate scaler
        if self.method == 'standard':
            self.scaler = StandardScaler(**self.scaler_kwargs)
        elif self.method == 'robust':
            self.scaler = RobustScaler(**self.scaler_kwargs)
        elif self.method == 'power':
            kwargs = {'method': 'yeo-johnson', 'standardize': True}
            kwargs.update(self.scaler_kwargs)
            self.scaler = PowerTransformer(**kwargs)
        elif self.method == 'quantile':
            kwargs = {'output_distribution': 'normal', 'random_state': 42}
            kwargs.update(self.scaler_kwargs)
            self.scaler = QuantileTransformer(**kwargs)
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler(**self.scaler_kwargs)

        self.scaler.fit(features)
        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted normalizer."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        if self.scaler is None:
            return features

        try:
            normalized = self.scaler.transform(features)
            return self._ensure_float32(normalized)
        except Exception as e:
            print(f"Warning: Normalization failed ({e}), returning original features")
            return features

    def get_name(self) -> str:
        return f"Feature Normalization ({self.method})"

    def get_params(self) -> dict:
        params = {'method': self.method}
        if self.scaler is not None:
            params.update(self.scaler.get_params())
        return params

    def get_feature_statistics(self) -> Optional[dict]:
        if self.scaler is None:
            return None

        stats = {'method': self.method}
        if hasattr(self.scaler, 'mean_'):
            stats['mean'] = self.scaler.mean_
        if hasattr(self.scaler, 'scale_'):
            stats['scale'] = self.scaler.scale_
        if hasattr(self.scaler, 'center_'):
            stats['center'] = self.scaler.center_
        if hasattr(self.scaler, 'quantiles_'):
            stats['quantiles'] = self.scaler.quantiles_
        return stats


class RobustFeatureNormalizer(FeatureNormalizer):
    def __init__(self, quantile_range: tuple = (25.0, 75.0), **kwargs):
        super().__init__(method='robust', quantile_range=quantile_range, **kwargs)

    def get_name(self) -> str:
        return "Robust Feature Normalization"


class PowerFeatureNormalizer(FeatureNormalizer):
    def __init__(self, power_method: str = 'yeo-johnson', standardize: bool = True, **kwargs):
        super().__init__(method='power', power_method=power_method, standardize=standardize, **kwargs)

    def get_name(self) -> str:
        return f"Power Feature Normalization ({self.scaler_kwargs.get('power_method', 'yeo-johnson')})"


class QuantileFeatureNormalizer(FeatureNormalizer):
    def __init__(self, output_distribution: str = 'normal',
                 n_quantiles: int = 1000, random_state: int = 42, **kwargs):
        super().__init__(
            method='quantile',
            output_distribution=output_distribution,
            n_quantiles=n_quantiles,
            random_state=random_state,
            **kwargs
        )

    def get_name(self) -> str:
        return f"Quantile Feature Normalization ({self.scaler_kwargs.get('output_distribution', 'normal')})"


class AdaptiveFeatureNormalizer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Adaptive normalizer that chooses the best method based on data characteristics."""

    def __init__(self, candidate_methods: list = None, selection_metric: str = 'kl_divergence'):
        """
        Initialize adaptive normalizer.

        Args:
            candidate_methods: List of normalization methods to try
            selection_metric: Metric for selecting best method
        """
        if candidate_methods is None:
            candidate_methods = ['robust', 'power', 'quantile']

        self.candidate_methods = candidate_methods
        self.selection_metric = selection_metric
        self.best_method = None
        self.best_normalizer = None
        self.method_scores = {}

    def _evaluate_normalization(self, original: np.ndarray, normalized: np.ndarray) -> float:
        """Evaluate the quality of normalization."""
        try:
            if self.selection_metric == 'kl_divergence':
                # Measure how close to normal distribution
                from scipy import stats

                # Test normality using Kolmogorov-Smirnov test
                # Lower p-value means less normal, so we want higher p-values
                _, p_value = stats.kstest(normalized.flatten(), 'norm')
                return p_value

            elif self.selection_metric == 'variance_ratio':
                # Measure variance stability across features
                feature_vars = np.var(normalized, axis=0)
                return 1.0 / (1.0 + np.std(feature_vars))

            elif self.selection_metric == 'range_stability':
                # Measure how stable the ranges are across features
                feature_ranges = np.ptp(normalized, axis=0)  # peak-to-peak
                return 1.0 / (1.0 + np.std(feature_ranges))

            else:
                # Default: use variance of standard deviations
                feature_stds = np.std(normalized, axis=0)
                return 1.0 / (1.0 + np.std(feature_stds))

        except Exception:
            return 0.0

    def _fit(self, features: np.ndarray) -> 'AdaptiveFeatureNormalizer':
        """Fit by testing different normalization methods."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        best_score = -np.inf

        for method in self.candidate_methods:
            try:
                # Test this normalization method
                normalizer = FeatureNormalizer(method=method)
                normalizer.fit(features)
                normalized = normalizer._transform(features)

                # Evaluate the normalization
                score = self._evaluate_normalization(features, normalized)
                self.method_scores[method] = score

                if score > best_score:
                    best_score = score
                    self.best_method = method
                    self.best_normalizer = normalizer

            except Exception as e:
                print(f"Warning: Method {method} failed: {e}")
                self.method_scores[method] = -np.inf

        if self.best_normalizer is None:
            # Fallback to robust scaling
            print("Warning: All normalization methods failed, using robust scaling")
            self.best_method = 'robust'
            self.best_normalizer = FeatureNormalizer(method='robust')
            self.best_normalizer.fit(features)

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform using the best normalization method."""
        if self.best_normalizer is None:
            raise ValueError("AdaptiveFeatureNormalizer must be fitted first")

        return self.best_normalizer._transform(features)

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        if self.best_method:
            return f"Adaptive Feature Normalization (selected: {self.best_method})"
        return "Adaptive Feature Normalization"

    def get_params(self) -> dict:
        """Get adaptive normalizer parameters."""
        params = {
            'candidate_methods': self.candidate_methods,
            'selection_metric': self.selection_metric,
            'best_method': self.best_method,
            'method_scores': self.method_scores
        }

        if self.best_normalizer is not None:
            params['best_normalizer_params'] = self.best_normalizer.get_params()

        return params