from typing import Optional, Tuple

import numpy as np

from preprocessing.feature.algorithm.adaboost import AdaBoostPipeline
from preprocessing.feature.algorithm.extratrees import ExtraTreesPipeline
from preprocessing.feature.algorithm.randomforest import RandomForestPipeline
from preprocessing.feature.algorithm.svm import SVMPipeline
from preprocessing.feature.algorithm.xgboost import XGBoostPipeline
from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline


class PreprocessingPipelineFactory:
    """Factory class to create appropriate preprocessing pipelines."""

    _pipelines = {
        'XGBoost': XGBoostPipeline,
        'RandomForest': RandomForestPipeline,
        'ExtraTrees': ExtraTreesPipeline,
        'AdaBoost': AdaBoostPipeline,
        'SVM': SVMPipeline
    }

    @classmethod
    def create_pipeline(cls, algorithm: str, **kwargs) -> AlgorithmPreprocessingPipeline:
        """Create a preprocessing pipeline for the specified algorithm."""
        if algorithm not in cls._pipelines:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                             f"Choose from {list(cls._pipelines.keys())}")

        return cls._pipelines[algorithm](**kwargs)

    @classmethod
    def register_pipeline(cls, algorithm: str, pipeline_class):
        """Register a new pipeline class."""
        cls._pipelines[algorithm] = pipeline_class

PIPELINE_CLASS_MAP = {
    cls.__name__: cls for cls in PreprocessingPipelineFactory._pipelines.values()
}


# Convenience functions for integration
def apply_feature_preprocessing(
        features: np.ndarray,
        labels: np.ndarray,
        algorithm: str,
        training: bool = True,
        save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, AlgorithmPreprocessingPipeline]:
    """
    Apply algorithm-specific preprocessing to features.

    Args:
        features: CNN-extracted features
        labels: Target labels
        algorithm: Target ML algorithm
        training: Whether this is training data
        save_path: Path to save fitted pipeline

    Returns:
        Tuple of (processed_features, processed_labels, pipeline)
    """
    import os

    # Load existing pipeline if provided
    if save_path and os.path.exists(save_path):
        # Dynamically load the correct pipeline class
        pipeline = AlgorithmPreprocessingPipeline.load(save_path)
        processed_features, processed_labels = pipeline.transform(
            features, labels, training=training
        )
        return processed_features, processed_labels, pipeline

    # Create new pipeline
    pipeline = PreprocessingPipelineFactory.create_pipeline(algorithm)

    if training:
        # Fit and transform training data
        pipeline.fit(features, labels)
        processed_features, processed_labels = pipeline.transform(
            features, labels, training=True
        )

        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pipeline.save(save_path)
    else:
        # For test data, just transform
        processed_features, processed_labels = pipeline.transform(
            features, labels, training=False
        )

    return processed_features, processed_labels, pipeline