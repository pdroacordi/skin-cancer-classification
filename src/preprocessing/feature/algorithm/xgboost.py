from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import SMOTEBalancing
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep


class XGBoostPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for XGBoost."""

    def _configure_pipeline(self):
        # 1. Remove low variance features
        self.steps.append(VarianceThresholdStep(threshold=1e-6))

        # 2. Feature selection with mutual information
        self.steps.append(FeatureSelectionStep(
            method='mutual_info',
            percentile=75
        ))

        # 3. Balancing with K-Means SMOTE (best F1: 0.861)
        self.balancing_strategy = SMOTEBalancing(
            method='kmeans_smote',
            k_neighbors=5,
            cluster_balance_threshold=0.1
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()