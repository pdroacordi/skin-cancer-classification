from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import SMOTEBalancing, TargetedSMOTEBalancing
from preprocessing.feature.steps.dimensionality_reduction import DimensionalityReductionStep
from preprocessing.feature.steps.normalization import NormalizationStep
from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep


class XGBoostPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for XGBoost."""

    def _configure_pipeline(self):
        self.steps.append(VarianceThresholdStep(threshold=0))

        self.steps.append(FeatureSelectionStep(
            method='importance_based',
            percentile=95  # Very conservative - keep 95% of CNN features
        ))

        self.balancing_strategy = TargetedSMOTEBalancing(
            minority_threshold=2000,  # Only oversample classes with < 2000 samples
            k_neighbors=3,  # Lower k for very small classes
            random_state=42
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()