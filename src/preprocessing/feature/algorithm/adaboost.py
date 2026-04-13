from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import ClassWeightBalancing
from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep

class AdaBoostPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for AdaBoost."""

    def _configure_pipeline(self):
        # 1. Soft outlier handling instead of removal
        self.steps.append(OutlierRemovalStep(contamination=0.05))

        # 2. Remove zero variance features
        self.steps.append(VarianceThresholdStep(threshold=0))

        # 3. Conservative feature selection
        self.steps.append(FeatureSelectionStep(
            method='f_score',
            percentile=85  # Keep 85% instead of 80%
        ))

        # 4. Use class weights instead of SMOTE
        self.balancing_strategy = ClassWeightBalancing()

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()
