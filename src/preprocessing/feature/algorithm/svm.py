from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import ClassWeightBalancing
from preprocessing.feature.steps.dimensionality_reduction import DimensionalityReductionStep
from preprocessing.feature.steps.normalization import NormalizationStep
from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep

class SVMPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for SVM."""

    def _configure_pipeline(self):
        # 1. Remove outliers (SVM is sensitive)
        self.steps.append(OutlierRemovalStep(
            method='isolation_forest',
            contamination=0.03
        ))

        # 2. Remove low variance features
        self.steps.append(VarianceThresholdStep(threshold=1e-6))

        # 3. ESSENTIAL: Normalize features
        self.steps.append(NormalizationStep(method='standard'))

        # 4. Feature selection
        self.steps.append(FeatureSelectionStep(
            method='mutual_info',
            percentile=60
        ))

        # 5. PCA for dimensionality reduction
        self.steps.append(DimensionalityReductionStep(
            method='pca',
            variance_threshold=0.95
        ))

        # 6. Use class weights for balancing
        self.balancing_strategy = ClassWeightBalancing()

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()