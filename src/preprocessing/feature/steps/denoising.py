import numpy as np

from ..base.preprocessor import (
    UnsupervisedFeaturePreprocessor, FeatureTransformationMixin
)


class FeatureDenoiser(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Remove noise from features using various methods."""

    def __init__(self, method: str = 'variance_threshold',
                 variance_threshold: float = 1e-6,
                 correlation_threshold: float = 0.95):
        """
        Initialize feature denoiser.

        Args:
            method: Denoising method ('variance_threshold', 'correlation_filter')
            variance_threshold: Threshold for removing low-variance features
            correlation_threshold: Threshold for removing highly correlated features
        """
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.feature_variances = None
        self.correlation_matrix = None

    def _fit(self, features: np.ndarray) -> 'FeatureDenoiser':
        """Fit the denoiser."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        n_features = features.shape[1]
        self.selected_features = np.ones(n_features, dtype=bool)

        if self.method == 'variance_threshold':
            # Remove low-variance features
            self.feature_variances = np.var(features, axis=0)
            low_var_mask = self.feature_variances > self.variance_threshold
            self.selected_features &= low_var_mask

        elif self.method == 'correlation_filter':
            # Remove highly correlated features
            self.correlation_matrix = np.corrcoef(features.T)

            # Find pairs of highly correlated features
            corr_mask = np.ones(n_features, dtype=bool)
            for i in range(n_features):
                if not corr_mask[i]:
                    continue
                for j in range(i + 1, n_features):
                    if not corr_mask[j]:
                        continue
                    if abs(self.correlation_matrix[i, j]) > self.correlation_threshold:
                        # Remove the feature with lower variance
                        var_i = np.var(features[:, i])
                        var_j = np.var(features[:, j])
                        if var_i < var_j:
                            corr_mask[i] = False
                            break
                        else:
                            corr_mask[j] = False

            self.selected_features &= corr_mask

        elif self.method == 'combined':
            # Apply both variance and correlation filtering
            # Variance filtering
            self.feature_variances = np.var(features, axis=0)
            low_var_mask = self.feature_variances > self.variance_threshold
            self.selected_features &= low_var_mask

            # Correlation filtering on remaining features
            remaining_features = features[:, self.selected_features]
            if remaining_features.shape[1] > 1:
                self.correlation_matrix = np.corrcoef(remaining_features.T)
                remaining_indices = np.where(self.selected_features)[0]

                corr_mask = np.ones(len(remaining_indices), dtype=bool)
                for i in range(len(remaining_indices)):
                    if not corr_mask[i]:
                        continue
                    for j in range(i + 1, len(remaining_indices)):
                        if not corr_mask[j]:
                            continue
                        if abs(self.correlation_matrix[i, j]) > self.correlation_threshold:
                            var_i = np.var(remaining_features[:, i])
                            var_j = np.var(remaining_features[:, j])
                            if var_i < var_j:
                                corr_mask[i] = False
                                break
                            else:
                                corr_mask[j] = False

                # Update selected features
                final_mask = np.zeros(n_features, dtype=bool)
                final_mask[remaining_indices[corr_mask]] = True
                self.selected_features = final_mask

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform by removing noisy features."""
        self._validate_features(features)

        if self.selected_features is None:
            return self._ensure_float32(features)

        return self._ensure_float32(features[:, self.selected_features])

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Feature Denoising ({self.method})"

    def get_params(self) -> dict:
        """Get denoiser parameters."""
        return {
            'method': self.method,
            'variance_threshold': self.variance_threshold,
            'correlation_threshold': self.correlation_threshold,
            'n_features_removed': np.sum(~self.selected_features) if self.selected_features is not None else None,
            'n_features_selected': np.sum(self.selected_features) if self.selected_features is not None else None
        }