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

    # 1) Se estamos em modo inferência E existe pipeline salvo, carregue-o:
    if not training and save_path and os.path.exists(save_path):
        pipeline = AlgorithmPreprocessingPipeline.load(save_path)

    else:
        # 2) Caso contrário, (re)crie e ajuste o pipeline nos dados atuais:
        pipeline =  PreprocessingPipelineFactory.create_pipeline(algorithm)
        pipeline.fit(features, labels)

        # 3) Se for modo treino e save_path fornecido, salve para inferência futura
        if training and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pipeline.save(save_path)

    # 4) Transforme sempre com o pipeline adequado
    processed_features, processed_labels = pipeline.transform(
        features, labels, training=training
    )
    return processed_features, processed_labels, pipeline