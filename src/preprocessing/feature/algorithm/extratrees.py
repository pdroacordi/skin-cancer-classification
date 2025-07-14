from imblearn.over_sampling import SMOTE

from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import SMOTEBalancing, ClassWeightBalancing
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep


class ExtraTreesPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for ExtraTrees."""

    def _configure_pipeline(self):
        # 1. Remove low variance features
        self.steps.append(VarianceThresholdStep(threshold=0))

        # 2. Keep most features - ExtraTrees handles high dimensionality well
        self.steps.append(FeatureSelectionStep(
            method='importance_based',
            percentile=90
        ))

        # 3. Use class weights instead of SMOTE-ENN
        self.balancing_strategy = ClassWeightBalancing()

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()
