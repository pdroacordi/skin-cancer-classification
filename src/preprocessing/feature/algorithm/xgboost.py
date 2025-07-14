from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import ClassWeightBalancing
from preprocessing.feature.steps.selection import FeatureSelectionStep, CorrelationBasedSelection
from preprocessing.feature.steps.threshold import VarianceThresholdStep


class XGBoostPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for XGBoost."""

    def _configure_pipeline(self):
        self.steps.append(VarianceThresholdStep(threshold=0))

        # 2. Remove only highly correlated features (>0.95 correlation)
        self.steps.append(CorrelationBasedSelection(
            correlation_threshold=0.95
        ))

        # 3. Use targeted SMOTE only for very small classes
        from preprocessing.feature.steps.balancing import TargetedSMOTEBalancing
        self.balancing_strategy = TargetedSMOTEBalancing(
            minority_threshold=500,  # Only oversample classes with < 500 samples
            k_neighbors=3,
            random_state=42
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()