"""
Feature engineering steps following the same pattern as graphic/steps/
"""

from typing import List

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures

from ..base.preprocessor import (
    UnsupervisedFeaturePreprocessor, FeatureTransformationMixin
)


class StatisticalFeatureEngineer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Engineer statistical features from existing features."""

    def __init__(self, feature_types: List[str] = None):
        """
        Initialize statistical feature engineer.

        Args:
            feature_types: List of statistical features to compute
        """
        if feature_types is None:
            feature_types = ['mean', 'std', 'median', 'q25', 'q75', 'min', 'max',
                             'skewness', 'kurtosis', 'energy']

        self.feature_types = feature_types
        self.n_input_features = None

    def _fit(self, features: np.ndarray) -> 'StatisticalFeatureEngineer':
        """Fit the statistical feature engineer."""
        self._validate_features(features)
        self.n_input_features = features.shape[1]
        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform by adding statistical features."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        stat_features = []

        # Per-sample statistics
        if 'mean' in self.feature_types:
            stat_features.append(np.mean(features, axis=1, keepdims=True))

        if 'std' in self.feature_types:
            stat_features.append(np.std(features, axis=1, keepdims=True))

        if 'median' in self.feature_types:
            stat_features.append(np.median(features, axis=1, keepdims=True))

        if 'q25' in self.feature_types:
            stat_features.append(np.percentile(features, 25, axis=1, keepdims=True))

        if 'q75' in self.feature_types:
            stat_features.append(np.percentile(features, 75, axis=1, keepdims=True))

        if 'min' in self.feature_types:
            stat_features.append(np.min(features, axis=1, keepdims=True))

        if 'max' in self.feature_types:
            stat_features.append(np.max(features, axis=1, keepdims=True))

        if 'skewness' in self.feature_types:
            skew = stats.skew(features, axis=1, nan_policy='omit')
            skew = np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)
            stat_features.append(skew.reshape(-1, 1))

        if 'kurtosis' in self.feature_types:
            kurt = stats.kurtosis(features, axis=1, nan_policy='omit')
            kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)
            stat_features.append(kurt.reshape(-1, 1))

        if 'energy' in self.feature_types:
            energy = np.sum(features ** 2, axis=1, keepdims=True)
            stat_features.append(energy)

        if 'entropy' in self.feature_types:
            # Approximate entropy using histogram
            entropy_vals = []
            for i in range(features.shape[0]):
                hist, _ = np.histogram(features[i], bins=10, density=True)
                hist = hist + 1e-10  # Avoid log(0)
                entropy = -np.sum(hist * np.log(hist))
                entropy_vals.append(entropy)
            stat_features.append(np.array(entropy_vals).reshape(-1, 1))

        if 'range' in self.feature_types:
            feature_range = np.ptp(features, axis=1, keepdims=True)  # peak-to-peak
            stat_features.append(feature_range)

        if 'iqr' in self.feature_types:
            q75 = np.percentile(features, 75, axis=1, keepdims=True)
            q25 = np.percentile(features, 25, axis=1, keepdims=True)
            iqr = q75 - q25
            stat_features.append(iqr)

        # Combine original features with statistical features
        if stat_features:
            out = np.hstack(stat_features)
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            return np.zeros((features.shape[0], 1))

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Statistical Feature Engineering ({len(self.feature_types)} stats)"

    def get_params(self) -> dict:
        """Get engineer parameters."""
        return {
            'feature_types': self.feature_types,
            'n_input_features': self.n_input_features
        }


class InteractionFeatureEngineer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Engineer interaction features between existing features."""

    def __init__(self, max_features: int = 20, max_degree: int = 2,
                 interaction_types: List[str] = None):
        """
        Initialize interaction feature engineer.

        Args:
            max_features: Maximum number of top features to use for interactions
            max_degree: Maximum degree of interactions
            interaction_types: Types of interactions to compute
        """
        if interaction_types is None:
            interaction_types = ['multiply', 'ratio', 'difference']

        self.max_features = max_features
        self.max_degree = max_degree
        self.interaction_types = interaction_types
        self.top_feature_indices = None

    def _fit(self, features: np.ndarray) -> 'InteractionFeatureEngineer':
        """Fit by selecting top features for interactions."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        # Select top features by variance for interactions
        feature_variances = np.var(features, axis=0)
        self.top_feature_indices = np.argsort(feature_variances)[-self.max_features:]

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform by adding interaction features."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        if self.top_feature_indices is None:
            return features

        top_features = features[:, self.top_feature_indices]
        interaction_features = []

        # Pairwise interactions
        for i in range(len(self.top_feature_indices)):
            for j in range(i + 1, min(i + 5, len(self.top_feature_indices))):  # Limit combinations
                feat_i = top_features[:, i]
                feat_j = top_features[:, j]

                if 'multiply' in self.interaction_types:
                    interaction_features.append((feat_i * feat_j).reshape(-1, 1))

                if 'ratio' in self.interaction_types:
                    # Avoid division by zero
                    ratio = feat_i / (feat_j + 1e-8)
                    ratio = np.clip(ratio, -1e6, 1e6)  # Clip extreme values
                    interaction_features.append(ratio.reshape(-1, 1))

                if 'difference' in self.interaction_types:
                    diff = feat_i - feat_j
                    interaction_features.append(diff.reshape(-1, 1))

                if 'sum' in self.interaction_types:
                    sum_feat = feat_i + feat_j
                    interaction_features.append(sum_feat.reshape(-1, 1))

        # Combine original features with interaction features
        if interaction_features:
            all_interaction_features = np.hstack(interaction_features)
            return np.hstack([features, all_interaction_features])
        else:
            return features

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Interaction Feature Engineering ({len(self.interaction_types)} types)"

    def get_params(self) -> dict:
        """Get engineer parameters."""
        return {
            'max_features': self.max_features,
            'max_degree': self.max_degree,
            'interaction_types': self.interaction_types,
            'n_top_features': len(self.top_feature_indices) if self.top_feature_indices is not None else 0
        }


class DomainFeatureEngineer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Engineer domain-specific features for medical imaging."""

    def __init__(self, feature_types: List[str] = None):
        """
        Initialize domain feature engineer.

        Args:
            feature_types: Types of domain features to compute
        """
        if feature_types is None:
            feature_types = ['texture_features', 'color_features', 'asymmetry_features',
                             'intensity_ratios', 'shape_features']

        self.feature_types = feature_types
        self.n_input_features = None

    def _fit(self, features: np.ndarray) -> 'DomainFeatureEngineer':
        """Fit the domain feature engineer."""
        self._validate_features(features)
        self.n_input_features = features.shape[1]
        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform by adding domain-specific features."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        domain_features = []

        # Texture-like features (assuming CNN features capture texture info)
        if 'texture_features' in self.feature_types:
            texture_feats = self._compute_texture_features(features)
            domain_features.append(texture_feats)

        # Color distribution features
        if 'color_features' in self.feature_types:
            color_feats = self._compute_color_features(features)
            domain_features.append(color_feats)

        # Asymmetry features (medical relevance)
        if 'asymmetry_features' in self.feature_types:
            asymmetry_feats = self._compute_asymmetry_features(features)
            domain_features.append(asymmetry_feats)

        # Intensity ratio features
        if 'intensity_ratios' in self.feature_types:
            ratio_feats = self._compute_intensity_ratios(features)
            domain_features.append(ratio_feats)

        # Shape-like features
        if 'shape_features' in self.feature_types:
            shape_feats = self._compute_shape_features(features)
            domain_features.append(shape_feats)

        # Combine original features with domain features
        if domain_features:
            all_domain_features = np.hstack(domain_features)
            all_domain_features = np.nan_to_num(all_domain_features,
                                                nan = 0.0,
                                                posinf = 1e6,
                                                neginf = -1e6)
            all_domain_features = np.clip(all_domain_features, -1e6, 1e6)
            return np.hstack([features, all_domain_features])
        else:
            return features

    def _compute_texture_features(self, features: np.ndarray) -> np.ndarray:
        """Compute texture-like features."""
        texture_features = []

        # Local Binary Pattern-like features using feature neighborhoods
        for i in range(min(8, features.shape[1])):
            center = features[:, i]
            start_idx = max(0, i - 1)
            end_idx = min(features.shape[1], i + 2)
            neighbors = features[:, start_idx:end_idx]

            if neighbors.shape[1] > 1:
                lbp_like = np.sum(neighbors > center.reshape(-1, 1), axis=1)
                texture_features.append(lbp_like.reshape(-1, 1))

        # Contrast measures
        if features.shape[1] >= 4:
            # Local contrast
            for i in range(0, min(features.shape[1] - 1, 10), 2):
                contrast = np.abs(features[:, i] - features[:, i + 1])
                texture_features.append(contrast.reshape(-1, 1))

        return np.hstack(texture_features) if texture_features else np.zeros((features.shape[0], 1))

    def _compute_color_features(self, features: np.ndarray) -> np.ndarray:
        """Compute color distribution features."""
        color_features = []

        # RGB-like channel features (assuming features represent color info)
        for i in range(0, min(features.shape[1], 12), 3):
            if i + 2 < features.shape[1]:
                r, g, b = features[:, i], features[:, i + 1], features[:, i + 2]
                total_intensity = r + g + b + 1e-8

                # Color ratios
                color_features.append((r / total_intensity).reshape(-1, 1))
                color_features.append((g / total_intensity).reshape(-1, 1))
                color_features.append((b / total_intensity).reshape(-1, 1))

                # Color distances
                rg_dist = np.abs(r - g).reshape(-1, 1)
                rb_dist = np.abs(r - b).reshape(-1, 1)
                gb_dist = np.abs(g - b).reshape(-1, 1)

                color_features.extend([rg_dist, rb_dist, gb_dist])

        return np.hstack(color_features) if color_features else np.zeros((features.shape[0], 1))

    def _compute_asymmetry_features(self, features: np.ndarray) -> np.ndarray:
        """Compute asymmetry features (medically relevant)."""
        asymmetry_features = []

        # Create asymmetry measures
        if features.shape[1] >= 4:
            mid_point = features.shape[1] // 2
            left_half = features[:, :mid_point]
            right_half = features[:, -mid_point:]

            if left_half.shape[1] == right_half.shape[1]:
                # Mirror asymmetry
                asymmetry = np.mean(np.abs(left_half - right_half), axis=1)
                asymmetry_features.append(asymmetry.reshape(-1, 1))

                # Correlation asymmetry
                corr_asymmetry = []
                for i in range(features.shape[0]):
                    corr = np.corrcoef(left_half[i], right_half[i])[0, 1]
                    corr_asymmetry.append(1 - corr if not np.isnan(corr) else 0)
                asymmetry_features.append(np.array(corr_asymmetry).reshape(-1, 1))

        # Radial asymmetry (quarters)
        if features.shape[1] >= 8:
            quarter_size = features.shape[1] // 4
            quarters = [
                features[:, i * quarter_size:(i + 1) * quarter_size]
                for i in range(4)
            ]

            # Compare opposite quarters
            if all(q.shape[1] == quarter_size for q in quarters):
                q1_q3_diff = np.mean(np.abs(quarters[0] - quarters[2]), axis=1)
                q2_q4_diff = np.mean(np.abs(quarters[1] - quarters[3]), axis=1)

                asymmetry_features.extend([
                    q1_q3_diff.reshape(-1, 1),
                    q2_q4_diff.reshape(-1, 1)
                ])

        return np.hstack(asymmetry_features) if asymmetry_features else np.zeros((features.shape[0], 1))

    def _compute_intensity_ratios(self, features: np.ndarray) -> np.ndarray:
        """Compute intensity ratio features."""
        ratio_features = []

        # High/low intensity ratios
        for percentile in [10, 25, 75, 90]:
            threshold = np.percentile(features, percentile, axis=1, keepdims=True)
            high_ratio = np.mean(features > threshold, axis=1)
            ratio_features.append(high_ratio.reshape(-1, 1))

        # Central vs peripheral intensity
        if features.shape[1] >= 6:
            center_size = features.shape[1] // 3
            center_start = (features.shape[1] - center_size) // 2
            center_end = center_start + center_size

            center_features = features[:, center_start:center_end]
            peripheral_features = np.hstack([
                features[:, :center_start],
                features[:, center_end:]
            ])

            if peripheral_features.shape[1] > 0:
                center_mean = np.mean(center_features, axis=1)
                peripheral_mean = np.mean(peripheral_features, axis=1)

                center_peripheral_ratio = center_mean / (peripheral_mean + 1e-8)
                ratio_features.append(center_peripheral_ratio.reshape(-1, 1))

        return np.hstack(ratio_features) if ratio_features else np.zeros((features.shape[0], 1))

    def _compute_shape_features(self, features: np.ndarray) -> np.ndarray:
        """Compute shape-like features."""
        shape_features = []

        # Compactness measures
        feature_sum = np.sum(features, axis=1)
        feature_sum_sq = np.sum(features ** 2, axis=1)
        compactness = feature_sum ** 2 / (feature_sum_sq + 1e-8)
        shape_features.append(compactness.reshape(-1, 1))

        # Elongation measures
        if features.shape[1] >= 2:
            # Principal component-like analysis
            for i in range(min(3, features.shape[1])):
                start_idx = i * (features.shape[1] // 3)
                end_idx = (i + 1) * (features.shape[1] // 3)
                segment = features[:, start_idx:end_idx]

                if segment.shape[1] > 0:
                    segment_var = np.var(segment, axis=1)
                    shape_features.append(segment_var.reshape(-1, 1))

        # Regularity measures
        mean_val = np.mean(features, axis=1, keepdims=True)
        regularity = np.mean(np.abs(features - mean_val), axis=1)
        shape_features.append(regularity.reshape(-1, 1))

        return np.hstack(shape_features) if shape_features else np.zeros((features.shape[0], 1))

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Domain Feature Engineering ({len(self.feature_types)} types)"

    def get_params(self) -> dict:
        """Get engineer parameters."""
        return {
            'feature_types': self.feature_types,
            'n_input_features': self.n_input_features
        }


class PolynomialFeatureEngineer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Engineer polynomial features."""

    def __init__(self, degree: int = 2, interaction_only: bool = True,
                 include_bias: bool = False, max_input_features: int = 50):
        """
        Initialize polynomial feature engineer.

        Args:
            degree: Polynomial degree
            interaction_only: Only interaction features, no powers
            include_bias: Include bias column
            max_input_features: Limit input features to prevent explosion
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_input_features = max_input_features
        self.poly_features = None
        self.selected_features = None

    def _fit(self, features: np.ndarray) -> 'PolynomialFeatureEngineer':
        """Fit the polynomial feature engineer."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        # Limit features to prevent combinatorial explosion
        if features.shape[1] > self.max_input_features:
            # Select top features by variance
            feature_vars = np.var(features, axis=0)
            top_indices = np.argsort(feature_vars)[-self.max_input_features:]
            self.selected_features = top_indices
            features_for_poly = features[:, top_indices]
        else:
            features_for_poly = features
            self.selected_features = None

        self.poly_features = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )

        self.poly_features.fit(features_for_poly)

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform by adding polynomial features."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        if self.poly_features is None:
            return features

        # Select same features used during fitting
        if self.selected_features is not None:
            features_for_poly = features[:, self.selected_features]
        else:
            features_for_poly = features

        try:
            poly_feats = self.poly_features.transform(features_for_poly)
            poly_feats = self._ensure_float32(poly_feats)

            # Combine original with polynomial features
            return np.hstack([features, poly_feats])

        except Exception as e:
            print(f"Warning: Polynomial feature generation failed: {e}")
            return features

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Polynomial Feature Engineering (degree {self.degree})"

    def get_params(self) -> dict:
        """Get engineer parameters."""
        return {
            'degree': self.degree,
            'interaction_only': self.interaction_only,
            'include_bias': self.include_bias,
            'max_input_features': self.max_input_features,
            'n_selected_features': len(self.selected_features) if self.selected_features is not None else None
        }


class FeatureClusterEngineer(UnsupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Engineer cluster-based features."""

    def __init__(self, n_clusters: int = 50, clustering_method: str = 'kmeans'):
        """
        Initialize feature cluster engineer.

        Args:
            n_clusters: Number of feature clusters
            clustering_method: Clustering method ('kmeans', 'hierarchical')
        """
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.feature_clusterer = None
        self.cluster_labels = None

    def _fit(self, features: np.ndarray) -> 'FeatureClusterEngineer':
        """Fit by clustering features."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        if features.shape[1] <= self.n_clusters:
            print(f"Warning: Number of features ({features.shape[1]}) <= n_clusters ({self.n_clusters})")
            self.cluster_labels = np.arange(features.shape[1])
            return self

        if self.clustering_method == 'kmeans':
            # Cluster features based on correlation
            corr_matrix = np.corrcoef(features.T)
            corr_distance = 1 - np.abs(corr_matrix)

            # Replace NaN values with 1 (maximum distance)
            corr_distance = np.nan_to_num(corr_distance, nan=1.0)

            # Use KMeans on correlation distances
            self.feature_clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )

            self.cluster_labels = self.feature_clusterer.fit_predict(corr_distance)

        elif self.clustering_method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering

            # Compute correlation matrix
            corr_matrix = np.corrcoef(features.T)
            corr_distance = 1 - np.abs(corr_matrix)
            corr_distance = np.nan_to_num(corr_distance, nan=1.0)

            self.feature_clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )

            self.cluster_labels = self.feature_clusterer.fit_predict(corr_distance)

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform by adding cluster-based features."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        if self.cluster_labels is None:
            return features

        cluster_features = []

        # Create cluster-based features
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id

            if np.sum(cluster_mask) > 0:
                cluster_data = features[:, cluster_mask]

                # Cluster statistics
                cluster_mean = np.mean(cluster_data, axis=1)
                cluster_max = np.max(cluster_data, axis=1)
                cluster_std = np.std(cluster_data, axis=1)

                cluster_features.extend([cluster_mean, cluster_max, cluster_std])

        if cluster_features:
            all_cluster_features = np.column_stack(cluster_features)
            return np.hstack([features, all_cluster_features])
        else:
            return features

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Feature Cluster Engineering ({self.clustering_method})"

    def get_params(self) -> dict:
        """Get engineer parameters."""
        return {
            'n_clusters': self.n_clusters,
            'clustering_method': self.clustering_method,
            'actual_clusters': len(np.unique(self.cluster_labels)) if self.cluster_labels is not None else None
        }