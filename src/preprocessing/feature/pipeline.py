"""
Main feature preprocessing pipeline following the same pattern as graphic/pipeline.py
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Dict, Any

import joblib
import numpy as np

from .base.preprocessor import FeaturePreprocessor
from .config import FeaturePreprocessingConfig, FeaturePreprocessingPresets
from .steps.engineering import (
    StatisticalFeatureEngineer, InteractionFeatureEngineer,
    DomainFeatureEngineer, PolynomialFeatureEngineer, FeatureClusterEngineer
)
from .steps.normalization import (
    FeatureNormalizer, AdaptiveFeatureNormalizer
)
from .steps.selection import (
    FeatureSelector, AdaptiveFeatureSelector
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePreprocessingPipeline:
    """Main feature preprocessing pipeline that orchestrates all preprocessing steps."""

    def __init__(self, config: FeaturePreprocessingConfig):
        """
        Initialize the feature preprocessing pipeline.

        Args:
            config: Feature preprocessing configuration
        """
        self.config = config
        self.steps: List[FeaturePreprocessor] = []
        self.step_names: List[str] = []
        self.is_fitted = False
        self.transformation_info: Dict[str, Any] = {}

        self._build_pipeline()

    def _build_pipeline(self):
        """Build the preprocessing pipeline based on configuration."""
        self.steps = []
        self.step_names = []

        # 1. Outlier detection (if enabled)
        if self.config.use_outlier_detection:
            from .steps.outlier_detection import OutlierDetector
            outlier_detector = OutlierDetector(
                method=self.config.outlier_method,
                contamination=self.config.outlier_contamination
            )
            self.steps.append(outlier_detector)
            self.step_names.append('outlier_detection')

        # 2. Feature denoising (if enabled)
        if self.config.use_feature_denoising:
            from .steps.denoising import FeatureDenoiser
            denoiser = FeatureDenoiser(
                method=self.config.denoising_method,
                variance_threshold=self.config.variance_threshold,
                correlation_threshold=self.config.correlation_threshold
            )
            self.steps.append(denoiser)
            self.step_names.append('feature_denoising')

        # 3. Normalization (if enabled)
        if self.config.use_normalization:
            if self.config.normalization_method == 'adaptive':
                normalizer = AdaptiveFeatureNormalizer()
            else:
                normalizer = FeatureNormalizer(method=self.config.normalization_method)

            self.steps.append(normalizer)
            self.step_names.append('normalization')

        # 4. Statistical feature engineering (if enabled)
        if self.config.use_statistical_features:
            stat_engineer = StatisticalFeatureEngineer(
                feature_types=self.config.statistical_feature_types
            )
            self.steps.append(stat_engineer)
            self.step_names.append('statistical_features')

        # 5. Domain feature engineering (if enabled)
        if self.config.use_domain_features:
            domain_engineer = DomainFeatureEngineer(
                feature_types=self.config.domain_feature_types
            )
            self.steps.append(domain_engineer)
            self.step_names.append('domain_features')

        # 6. Interaction feature engineering (if enabled)
        if self.config.use_interaction_features:
            interaction_engineer = InteractionFeatureEngineer(
                max_features=self.config.interaction_max_features,
                max_degree=self.config.interaction_max_degree
            )
            self.steps.append(interaction_engineer)
            self.step_names.append('interaction_features')

        # 7. Polynomial feature engineering (if enabled)
        if self.config.use_polynomial_features:
            poly_engineer = PolynomialFeatureEngineer(
                degree=self.config.polynomial_degree,
                interaction_only=self.config.polynomial_interaction_only
            )
            self.steps.append(poly_engineer)
            self.step_names.append('polynomial_features')

        # 8. Feature clustering (if enabled)
        if self.config.use_feature_clustering:
            cluster_engineer = FeatureClusterEngineer(
                n_clusters=self.config.n_feature_clusters,
                clustering_method=self.config.clustering_method
            )
            self.steps.append(cluster_engineer)
            self.step_names.append('feature_clustering')

        # 9. Dimensionality reduction - PCA (if enabled)
        if self.config.use_pca:
            from .steps.dimensionality_reduction import PCAReducer
            pca_reducer = PCAReducer(
                variance_threshold=self.config.pca_variance_threshold,
                max_components=self.config.pca_max_components
            )
            self.steps.append(pca_reducer)
            self.step_names.append('pca')

        # 10. Dimensionality reduction - Kernel PCA (if enabled)
        if self.config.use_kernel_pca:
            from .steps.dimensionality_reduction import KernelPCAReducer
            kernel_pca_reducer = KernelPCAReducer(
                n_components=self.config.kernel_pca_components,
                kernel=self.config.kernel_pca_kernel,
                gamma=self.config.kernel_pca_gamma
            )
            self.steps.append(kernel_pca_reducer)
            self.step_names.append('kernel_pca')

        # 11. Feature selection (if enabled) - should be last
        if self.config.use_feature_selection:
            if self.config.selection_method == 'adaptive':
                selector = AdaptiveFeatureSelector(
                    k=self.config.selection_k
                )
            else:
                selector = FeatureSelector(
                    method=self.config.selection_method,
                    k=self.config.selection_k,
                    percentile=self.config.selection_percentile
                )

            self.steps.append(selector)
            self.step_names.append('feature_selection')

        if self.config.verbose:
            enabled_steps = [name for name in self.step_names]
            logger.info(f"Built pipeline with {len(self.steps)} steps: {enabled_steps}")

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> 'FeaturePreprocessingPipeline':
        """
        Fit the preprocessing pipeline to the training data.

        Args:
            features: Training feature matrix of shape (n_samples, n_features)
            labels: Training labels of shape (n_samples,)

        Returns:
            self: Fitted pipeline
        """
        if self.config.verbose:
            logger.info(f"Fitting feature preprocessing pipeline on data: {features.shape}")

        # Validate input
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        if features.ndim != 2:
            raise ValueError(f"Features must be 2D, got {features.ndim}D")

        original_shape = features.shape
        current_features = features.copy()

        # Fit each step
        for i, (step, step_name) in enumerate(zip(self.steps, self.step_names)):
            if self.config.verbose:
                logger.info(f"Fitting step {i + 1}/{len(self.steps)}: {step.get_name()}")

            try:
                # Fit the step
                step.fit(current_features, labels)

                # Transform to get the output for next step
                current_features = step.process(current_features, labels)

                # Store transformation info
                self.transformation_info[step_name] = {
                    'step_name': step.get_name(),
                    'input_shape': current_features.shape if i == 0 else
                    self.transformation_info[self.step_names[i - 1]]['output_shape'],
                    'output_shape': current_features.shape,
                    'params': step.get_params() if hasattr(step, 'get_params') else {}
                }

                if self.config.verbose:
                    logger.info(
                        f"  {step.get_name()}: {self.transformation_info[step_name]['input_shape']} -> {self.transformation_info[step_name]['output_shape']}")

            except Exception as e:
                logger.error(f"Error in step {step_name}: {e}")
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
                # Continue with next step
                continue

        # Store final transformation info
        final_shape = current_features.shape
        self.transformation_info['pipeline'] = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'compression_ratio': final_shape[1] / original_shape[1] if original_shape[1] > 0 else 0,
            'n_samples': original_shape[0],
            'steps_completed': len([s for s in self.transformation_info.keys() if s != 'pipeline'])
        }

        self.is_fitted = True

        if self.config.verbose:
            logger.info(f"Pipeline fitted successfully: {original_shape} -> {final_shape}")
            logger.info(f"Compression ratio: {self.transformation_info['pipeline']['compression_ratio']:.3f}")

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted pipeline.

        Args:
            features: Feature matrix to transform

        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming")

        if not isinstance(features, np.ndarray):
            features = np.array(features)

        current_features = features.copy()

        # Apply each step
        for step, step_name in zip(self.steps, self.step_names):
            try:
                current_features = step.process(current_features)
            except Exception as e:
                logger.warning(f"Error in transform step {step_name}: {e}")
                # Continue with current features
                continue

        return current_features

    def fit_transform(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the pipeline and transform the features in one step.

        Args:
            features: Training feature matrix
            labels: Training labels

        Returns:
            Transformed features
        """
        return self.fit(features, labels).transform(features)

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the feature transformation."""
        if not self.is_fitted:
            return {'fitted': False}

        return {
            'fitted': True,
            'pipeline_info': self.transformation_info,
            'config': self.config.to_dict(),
            'steps': [
                {
                    'name': step.get_name(),
                    'params': step.get_params() if hasattr(step, 'get_params') else {}
                }
                for step in self.steps
            ]
        }

    def save(self, filepath: str) -> None:
        """Save the fitted pipeline to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")

        save_data = {
            'pipeline': self,
            'config': self.config,
            'transformation_info': self.transformation_info,
            'is_fitted': self.is_fitted
        }

        joblib.dump(save_data, filepath)

        if self.config.verbose:
            logger.info(f"Pipeline saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FeaturePreprocessingPipeline':
        """Load a fitted pipeline from disk."""
        save_data = joblib.load(filepath)

        pipeline = save_data['pipeline']
        pipeline.config = save_data['config']
        pipeline.transformation_info = save_data['transformation_info']
        pipeline.is_fitted = save_data['is_fitted']

        logger.info(f"Pipeline loaded from: {filepath}")
        return pipeline

    def get_step_by_name(self, step_name: str) -> Optional[FeaturePreprocessor]:
        """Get a specific step by name."""
        for step, name in zip(self.steps, self.step_names):
            if name == step_name:
                return step
        return None

    def visualize_pipeline(self) -> None:
        """Print a visualization of the pipeline."""
        if not self.is_fitted:
            print("Pipeline not fitted yet")
            return

        print("=" * 60)
        print("FEATURE PREPROCESSING PIPELINE")
        print("=" * 60)

        total_info = self.transformation_info.get('pipeline', {})
        if total_info:
            print(f"Input shape: {total_info['original_shape']}")
            print(f"Output shape: {total_info['final_shape']}")
            print(f"Compression ratio: {total_info['compression_ratio']:.3f}")
            print()

        for i, (step, step_name) in enumerate(zip(self.steps, self.step_names)):
            step_info = self.transformation_info.get(step_name, {})
            input_shape = step_info.get('input_shape', 'Unknown')
            output_shape = step_info.get('output_shape', 'Unknown')

            print(f"{i + 1:2d}. {step.get_name()}")
            print(f"    {input_shape} -> {output_shape}")

            # Show key parameters
            params = step_info.get('params', {})
            if params:
                key_params = {k: v for k, v in params.items()
                              if k in ['method', 'k', 'n_clusters', 'degree'] and v is not None}
                if key_params:
                    param_str = ', '.join([f"{k}={v}" for k, v in key_params.items()])
                    print(f"    Parameters: {param_str}")
            print()

        print("=" * 60)


# Convenience function matching the graphic preprocessing API
def apply_feature_preprocessing(features: np.ndarray,
                                labels: Optional[np.ndarray] = None,
                                config: Optional[FeaturePreprocessingConfig] = None,
                                preset: str = 'medical_imaging',
                                training: bool = True,
                                save_path: Optional[str] = None) -> tuple:
    """
    Apply feature preprocessing to CNN-extracted features.

    Args:
        features: Input feature matrix (n_samples, n_features)
        labels: Target labels (required for supervised preprocessing)
        config: Custom preprocessing configuration
        preset: Preset configuration ('basic', 'standard', 'advanced', 'medical_imaging', 'performance')
        training: Whether to fit the model
        save_path: Path to save the fitted pipeline

    Returns:
        tuple: (processed_features, fitted_pipeline)
    """
    if not training and save_path and os.path.exists(save_path):
        pipeline = FeaturePreprocessingPipeline.load(save_path)
        return pipeline.transform(features), pipeline

    # Get configuration
    if config is None:
        if preset == 'basic':
            config = FeaturePreprocessingPresets.get_basic_config()
        elif preset == 'standard':
            config = FeaturePreprocessingPresets.get_standard_config()
        elif preset == 'advanced':
            config = FeaturePreprocessingPresets.get_advanced_config()
        elif preset == 'medical_imaging':
            config = FeaturePreprocessingPresets.get_medical_imaging_config()
        elif preset == 'performance':
            config = FeaturePreprocessingPresets.get_performance_config()
        else:
            raise ValueError(f"Unknown preset: {preset}")

    # Create and fit pipeline
    pipeline = FeaturePreprocessingPipeline(config)
    if training:
        processed_features = pipeline.fit_transform(features, labels)
    else:
        processed_features = pipeline.transform(features)

    # Save pipeline if requested
    if save_path:
        pipeline.save(save_path)

    return processed_features, pipeline


# Factory functions for common use cases
def create_basic_pipeline() -> FeaturePreprocessingPipeline:
    """Create a basic feature preprocessing pipeline."""
    config = FeaturePreprocessingPresets.get_basic_config()
    return FeaturePreprocessingPipeline(config)


def create_medical_imaging_pipeline() -> FeaturePreprocessingPipeline:
    """Create a pipeline optimized for medical imaging features."""
    config = FeaturePreprocessingPresets.get_medical_imaging_config()
    return FeaturePreprocessingPipeline(config)


def create_performance_pipeline() -> FeaturePreprocessingPipeline:
    """Create a fast pipeline optimized for performance."""
    config = FeaturePreprocessingPresets.get_performance_config()
    return FeaturePreprocessingPipeline(config)


def create_custom_pipeline(**kwargs) -> FeaturePreprocessingPipeline:
    """Create a custom pipeline with specified parameters."""
    config = FeaturePreprocessingConfig(**kwargs)
    return FeaturePreprocessingPipeline(config)


# Integration helper for existing codebase
def enhance_cnn_features(features: np.ndarray,
                         labels: np.ndarray,
                         method: str = 'medical_imaging') -> tuple:
    """
    Enhance CNN features for better classical ML performance.

    This is the main function to integrate into existing pipelines.

    Args:
        features: CNN-extracted features
        labels: Target labels
        method: Enhancement method ('basic', 'standard', 'medical_imaging', 'advanced')

    Returns:
        tuple: (enhanced_features, fitted_pipeline)
    """
    return apply_feature_preprocessing(
        features=features,
        labels=labels,
        preset=method
    )