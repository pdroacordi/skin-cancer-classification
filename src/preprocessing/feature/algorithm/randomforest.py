from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import ClassWeightBalancing
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep

class RandomForestPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for Random Forest."""

    def _configure_pipeline(self):
        # 1. Remove low variance features
        self.steps.append(VarianceThresholdStep(threshold=1e-6))

        # 2. Feature selection with RFE
        self.steps.append(FeatureSelectionStep(
            method='rfe',
            percentile=80
        ))

        # 3. Use class weights for balancing
        self.balancing_strategy = ClassWeightBalancing()

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()

