"""
Configuration for feature preprocessing pipeline.
Follows the same pattern as graphic/config.py
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class FeaturePreprocessingConfig:
    """Configuration for feature preprocessing pipeline."""

    # Normalization settings
    use_normalization: bool = True
    normalization_method: str = 'robust'  # 'standard', 'robust', 'power', 'quantile'

    # Dimensionality reduction settings
    use_pca: bool = True
    pca_variance_threshold: float = 0.95
    pca_max_components: Optional[int] = None

    use_kernel_pca: bool = False
    kernel_pca_components: Optional[int] = None
    kernel_pca_kernel: str = 'rbf'
    kernel_pca_gamma: Optional[float] = None

    # Feature selection settings
    use_feature_selection: bool = True
    selection_method: str = 'mutual_info'  # 'f_score', 'mutual_info', 'rfe', 'lasso'
    selection_k: Optional[int] = None  # Auto-determined if None
    selection_percentile: Optional[float] = None  # Alternative to k

    # Feature engineering settings
    use_statistical_features: bool = True
    statistical_feature_types: list = None  # Auto-determined if None

    use_interaction_features: bool = True
    interaction_max_features: int = 20  # Limit interactions to prevent explosion
    interaction_max_degree: int = 2

    use_polynomial_features: bool = False
    polynomial_degree: int = 2
    polynomial_interaction_only: bool = True

    # Domain-specific features
    use_domain_features: bool = True
    domain_feature_types: list = None  # Auto-determined if None

    # Feature clustering
    use_feature_clustering: bool = True
    n_feature_clusters: int = 50
    clustering_method: str = 'kmeans'  # 'kmeans', 'hierarchical'
    clustering_linkage: str = 'ward'  # For hierarchical clustering

    # Outlier detection and noise reduction
    use_outlier_detection: bool = True
    outlier_method: str = 'isolation_forest'  # 'isolation_forest', 'local_outlier_factor', 'one_class_svm'
    outlier_contamination: float = 0.05

    use_feature_denoising: bool = True
    denoising_method: str = 'variance_threshold'  # 'variance_threshold', 'correlation_filter'
    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95

    # Advanced transformations
    use_log_transform: bool = False
    log_transform_features: list = None  # Specific features to log-transform

    use_box_cox: bool = False
    box_cox_lambda: Optional[float] = None  # Auto-determined if None

    # Ensemble settings
    use_ensemble_preprocessing: bool = False
    n_ensemble_methods: int = 3
    ensemble_weights: Optional[Dict[str, float]] = None

    # Performance settings
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True

    # Memory management
    batch_processing: bool = False
    batch_size: int = 1000
    use_sparse_matrices: bool = False

    # Validation and monitoring
    validate_transformations: bool = True
    monitor_feature_drift: bool = False
    drift_threshold: float = 0.1

    def __post_init__(self):
        """Post-initialization validation and defaults."""
        # Set default feature types if not specified
        if self.statistical_feature_types is None:
            self.statistical_feature_types = [
                'mean', 'std', 'median', 'q25', 'q75', 'min', 'max',
                'skewness', 'kurtosis', 'energy'
            ]

        if self.domain_feature_types is None:
            self.domain_feature_types = [
                'texture_features', 'color_features', 'asymmetry_features',
                'shape_features', 'intensity_ratios'
            ]

        # Validation
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if not 0 < self.pca_variance_threshold <= 1:
            raise ValueError("pca_variance_threshold must be between 0 and 1")

        if self.selection_k is not None and self.selection_k <= 0:
            raise ValueError("selection_k must be positive")

        if self.selection_percentile is not None:
            if not 0 < self.selection_percentile <= 100:
                raise ValueError("selection_percentile must be between 0 and 100")

        if not 0 < self.outlier_contamination < 0.5:
            raise ValueError("outlier_contamination must be between 0 and 0.5")

        if self.n_feature_clusters <= 0:
            raise ValueError("n_feature_clusters must be positive")

        if self.polynomial_degree < 1:
            raise ValueError("polynomial_degree must be at least 1")

        valid_normalization_methods = ['standard', 'robust', 'power', 'quantile', 'none']
        if self.normalization_method not in valid_normalization_methods:
            raise ValueError(f"normalization_method must be one of {valid_normalization_methods}")

        valid_selection_methods = ['f_score', 'mutual_info', 'rfe', 'lasso', 'univariate_chi2']
        if self.selection_method not in valid_selection_methods:
            raise ValueError(f"selection_method must be one of {valid_selection_methods}")

    def get_preprocessing_steps(self) -> list:
        """Get list of enabled preprocessing steps in order."""
        steps = []

        if self.use_outlier_detection:
            steps.append('outlier_detection')

        if self.use_feature_denoising:
            steps.append('feature_denoising')

        if self.use_normalization:
            steps.append('normalization')

        if self.use_statistical_features:
            steps.append('statistical_features')

        if self.use_domain_features:
            steps.append('domain_features')

        if self.use_interaction_features:
            steps.append('interaction_features')

        if self.use_polynomial_features:
            steps.append('polynomial_features')

        if self.use_feature_clustering:
            steps.append('feature_clustering')

        if self.use_pca:
            steps.append('pca')

        if self.use_kernel_pca:
            steps.append('kernel_pca')

        if self.use_feature_selection:
            steps.append('feature_selection')

        return steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeaturePreprocessingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def copy(self) -> 'FeaturePreprocessingConfig':
        """Create a copy of the configuration."""
        return self.from_dict(self.to_dict())

    def update(self, **kwargs) -> 'FeaturePreprocessingConfig':
        """Update configuration parameters."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)


# Predefined configurations for different use cases
class FeaturePreprocessingPresets:
    """Predefined preprocessing configurations for common scenarios."""

    @staticmethod
    def get_basic_config() -> FeaturePreprocessingConfig:
        """Basic preprocessing with essential steps."""
        return FeaturePreprocessingConfig(
            use_normalization=True,
            normalization_method='robust',
            use_pca=True,
            pca_variance_threshold=0.95,
            use_feature_selection=True,
            selection_method='f_score',
            use_statistical_features=False,
            use_interaction_features=False,
            use_domain_features=False,
            use_feature_clustering=False,
            use_outlier_detection=False,
            verbose=False
        )

    @staticmethod
    def get_standard_config() -> FeaturePreprocessingConfig:
        """Standard preprocessing with moderate feature engineering."""
        return FeaturePreprocessingConfig(
            use_normalization=True,
            normalization_method='robust',
            use_pca=True,
            pca_variance_threshold=0.95,
            use_feature_selection=True,
            selection_method='mutual_info',
            use_statistical_features=True,
            use_interaction_features=True,
            interaction_max_features=15,
            use_domain_features=True,
            use_feature_clustering=True,
            n_feature_clusters=30,
            use_outlier_detection=True,
            outlier_contamination=0.05
        )

    @staticmethod
    def get_advanced_config() -> FeaturePreprocessingConfig:
        """Advanced preprocessing with comprehensive feature engineering."""
        return FeaturePreprocessingConfig(
            use_normalization=True,
            normalization_method='power',
            use_pca=True,
            pca_variance_threshold=0.98,
            use_kernel_pca=True,
            kernel_pca_components=100,
            use_feature_selection=True,
            selection_method='mutual_info',
            use_statistical_features=True,
            use_interaction_features=True,
            interaction_max_features=20,
            use_polynomial_features=True,
            polynomial_degree=2,
            use_domain_features=True,
            use_feature_clustering=True,
            n_feature_clusters=50,
            use_outlier_detection=True,
            outlier_method='isolation_forest',
            use_feature_denoising=True,
            use_ensemble_preprocessing=True,
            n_ensemble_methods=3
        )

    @staticmethod
    def get_medical_imaging_config() -> FeaturePreprocessingConfig:
        """Specialized configuration for medical imaging features."""
        return FeaturePreprocessingConfig(
            use_normalization=True,
            normalization_method='robust',
            use_pca=True,
            pca_variance_threshold=0.95,
            use_feature_selection=True,
            selection_method='mutual_info',
            use_statistical_features=True,
            statistical_feature_types=[
                'mean', 'std', 'median', 'q25', 'q75', 'skewness', 'kurtosis'
            ],
            use_interaction_features=True,
            interaction_max_features=15,
            use_domain_features=True,
            domain_feature_types=[
                'texture_features', 'asymmetry_features', 'color_features',
                'intensity_ratios', 'shape_features'
            ],
            use_feature_clustering=True,
            n_feature_clusters=40,
            use_outlier_detection=True,
            outlier_method='isolation_forest',
            outlier_contamination=0.03,  # Lower for medical data
            use_feature_denoising=True,
            correlation_threshold=0.98,  # Higher for medical features
            verbose=True
        )

    @staticmethod
    def get_performance_config() -> FeaturePreprocessingConfig:
        """Fast processing configuration with minimal overhead."""
        return FeaturePreprocessingConfig(
            use_normalization=True,
            normalization_method='standard',
            use_pca=True,
            pca_variance_threshold=0.90,
            use_feature_selection=True,
            selection_method='f_score',
            selection_percentile=50,  # Keep top 50%
            use_statistical_features=False,
            use_interaction_features=False,
            use_domain_features=False,
            use_feature_clustering=False,
            use_outlier_detection=False,
            use_feature_denoising=True,
            variance_threshold=1e-4,
            verbose=False,
            batch_processing=True
        )