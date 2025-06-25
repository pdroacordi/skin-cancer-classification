import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List

import joblib
import numpy as np

from preprocessing.feature.base.balancing import BalancingStrategy
from preprocessing.feature.base.step import BasePreprocessingStep
from preprocessing.feature.steps.balancing import ClassWeightBalancing
from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlgorithmPreprocessingPipeline(ABC):
    """Abstract base class for algorithm-specific preprocessing pipelines."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.steps: List[BasePreprocessingStep] = []
        self.balancing_strategy: Optional[BalancingStrategy] = None
        self.is_fitted = False
        self.feature_stats = {}
        
    @abstractmethod
    def _configure_pipeline(self):
        """Configure preprocessing steps for the specific algorithm."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AlgorithmPreprocessingPipeline':
        """Fit the preprocessing pipeline."""
        logger.info(f"Fitting {self.__class__.__name__}")
        logger.info(f"Input shape: {X.shape}, Class distribution: {np.bincount(y)}")
        
        self.feature_stats['n_features_original'] = X.shape[1]
        self.feature_stats['n_samples_original'] = X.shape[0]
        
        # Apply each preprocessing step
        X_transformed = X.copy()
        y_transformed = y.copy()
        
        for i, step in enumerate(self.steps):
            logger.info(f"Fitting step {i+1}/{len(self.steps)}: {step.__class__.__name__}")
            
            if isinstance(step, OutlierRemovalStep):
                # Special handling for outlier removal
                step.fit(X_transformed, y_transformed)
                if hasattr(step, 'outlier_mask') and step.outlier_mask is not None:
                    X_transformed = X_transformed[step.outlier_mask]
                    y_transformed = y_transformed[step.outlier_mask]
            else:
                X_transformed = step.fit_transform(X_transformed, y_transformed)
            
            logger.info(f"  Shape after {step.__class__.__name__}: {X_transformed.shape}")
        
        self.feature_stats['n_features_final'] = X_transformed.shape[1]
        self.feature_stats['n_samples_after_outliers'] = X_transformed.shape[0]
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                  training: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform the data."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        X_transformed = X.copy()
        y_transformed = y.copy() if y is not None else None
        
        # Apply each step
        for step in self.steps:
            if isinstance(step, OutlierRemovalStep) and not training:
                # Don't remove outliers from test data
                continue
            X_transformed = step.transform(X_transformed)
        
        # Apply balancing only for training
        if training and y_transformed is not None and self.balancing_strategy:
            if not isinstance(self.balancing_strategy, ClassWeightBalancing):
                X_transformed, y_transformed = self.balancing_strategy.balance(
                    X_transformed, y_transformed
                )
                logger.info(f"Balanced data: {len(y)} -> {len(y_transformed)} samples")
                logger.info(f"New class distribution: {np.bincount(y_transformed)}")
        
        return X_transformed, y_transformed
    
    def get_params(self) -> Dict[str, Any]:
        """Get all pipeline parameters."""
        params = {
            'algorithm': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'feature_stats': self.feature_stats,
            'steps': [
                {
                    'name': step.__class__.__name__,
                    'params': step.get_params()
                }
                for step in self.steps
            ]
        }
        
        if self.balancing_strategy:
            params['balancing'] = self.balancing_strategy.get_params()
        
        return params
    
    def save(self, filepath: str):
        """Save the fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline")
        
        save_data = {
            'class': self.__class__.__name__,
            'steps': self.steps,
            'balancing_strategy': self.balancing_strategy,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AlgorithmPreprocessingPipeline':
        """Load a fitted pipeline."""
        save_data = joblib.load(filepath)
        class_name = save_data['class']

        from preprocessing.feature.pipeline import PIPELINE_CLASS_MAP
        if class_name not in PIPELINE_CLASS_MAP:
            raise ValueError(f"Pipeline class '{class_name}' not registered.")

        pipeline_class = PIPELINE_CLASS_MAP[class_name]
        pipeline = pipeline_class()

        pipeline.steps = save_data['steps']
        pipeline.balancing_strategy = save_data['balancing_strategy']
        pipeline.feature_stats = save_data['feature_stats']
        pipeline.is_fitted = save_data['is_fitted']
        pipeline.random_state = save_data['random_state']
        
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline