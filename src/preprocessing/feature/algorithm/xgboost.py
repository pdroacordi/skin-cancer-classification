from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import SMOTEBalancing
from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep


class XGBoostPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for XGBoost."""

    def _configure_pipeline(self):
        # 1. Remove outliers to improve robustness
        self.steps.append(OutlierRemovalStep(
            method='isolation_forest',
            contamination=0.01
        ))

        # 2. Remove low variance features
        self.steps.append(VarianceThresholdStep(threshold=1e-4))

        # 3. Feature selection via model-based importance
        self.steps.append(FeatureSelectionStep(
            method='importance_based',
            percentile=75
        ))

        # 4. Balancing with SMOTE (robust for small classes)
        self.balancing_strategy = SMOTEBalancing(
            method='kmeans_smote',
            k_neighbors=3,
            cluster_balance_threshold=0.01
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()