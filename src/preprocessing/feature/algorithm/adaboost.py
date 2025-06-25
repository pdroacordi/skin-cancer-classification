from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import SMOTEBalancing
from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep

class AdaBoostPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for AdaBoost."""

    def _configure_pipeline(self):
        # 1. Remove outliers (AdaBoost is VERY sensitive)
        self.steps.append(OutlierRemovalStep(
            method='isolation_forest',
            contamination=0.05
        ))

        # 2. Remove low variance features
        self.steps.append(VarianceThresholdStep(threshold=1e-5))

        # 3. Fast feature selection with F-score
        self.steps.append(FeatureSelectionStep(
            method='f_score',
            percentile=80
        ))

        # 4. Standard SMOTE for balancing
        self.balancing_strategy = SMOTEBalancing(
            method='smote',
            k_neighbors=5
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()
