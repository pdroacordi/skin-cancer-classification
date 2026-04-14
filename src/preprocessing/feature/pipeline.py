from typing import Optional, Tuple

import numpy as np

from preprocessing.feature.algorithm.configurable import ConfigurablePreprocessingPipeline
from preprocessing.feature.algorithm.configs import ALGORITHM_PIPELINE_CONFIGS
from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline


class PreprocessingPipelineFactory:
    """Factory class to create appropriate preprocessing pipelines."""

    @classmethod
    def create_pipeline(cls, algorithm: str, **kwargs) -> AlgorithmPreprocessingPipeline:
        """Create a preprocessing pipeline for the specified algorithm."""
        if algorithm not in ALGORITHM_PIPELINE_CONFIGS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Choose from {list(ALGORITHM_PIPELINE_CONFIGS)}"
            )
        return ConfigurablePreprocessingPipeline(algorithm, **kwargs)


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