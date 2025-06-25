from imblearn.over_sampling import SMOTE

from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import SMOTEBalancing
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep


class ExtraTreesPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for ExtraTrees."""

    def _configure_pipeline(self):
        # 1. Remove low variance features
        self.steps.append(VarianceThresholdStep(threshold=1e-6))

        # 2. Feature selection based on importance
        self.steps.append(FeatureSelectionStep(
            method='importance_based',
            percentile=70
        ))

        # 3. Hybrid balancing with SMOTE-ENN
        self.balancing_strategy = SMOTEBalancing(
            method='smote_enn',
            smote=SMOTE(k_neighbors=5, random_state=self.random_state)
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()
