from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import ClassWeightBalancing, TargetedSMOTEBalancing, SMOTEBalancing
from preprocessing.feature.steps.dimensionality_reduction import DimensionalityReductionStep
from preprocessing.feature.steps.normalization import NormalizationStep
from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep
from preprocessing.feature.steps.selection import FeatureSelectionStep, CorrelationBasedSelection
from preprocessing.feature.steps.threshold import VarianceThresholdStep


class XGBoostPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for XGBoost."""

    def _configure_pipeline(self):
        self.steps += [
            VarianceThresholdStep(0),
            OutlierRemovalStep(contamination=0.05),
            NormalizationStep('standard'),
            DimensionalityReductionStep('pca', 0.95),
            FeatureSelectionStep(method='mutual_info', percentile=90)
        ]
        # K-Means-SMOTE só para classes < 600
        self.balancing_strategy = SMOTEBalancing(
            method='smote_enn',  # melhor que smote puro
            sampling_strategy='not majority'
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()