from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import ClassWeightBalancing
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep

class RandomForestPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for Random Forest."""

    def _configure_pipeline(self):
        # ONLY remove exactly zero-variance features
        # These are truly useless and just slow down training
        self.steps.append(VarianceThresholdStep(threshold=0))

        # That's it! No feature selection - RF does this internally
        # No normalization - RF doesn't need it
        # No outlier removal - RF is robust to outliers

        # Use class weights for balancing
        self.balancing_strategy = ClassWeightBalancing()

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()

