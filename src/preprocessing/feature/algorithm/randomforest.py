from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import ClassWeightBalancing, SMOTEBalancing
from preprocessing.feature.steps.dimensionality_reduction import DimensionalityReductionStep
from preprocessing.feature.steps.normalization import NormalizationStep
from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep

class RandomForestPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for Random Forest."""

    def _configure_pipeline(self):
        self.steps += [
            VarianceThresholdStep(0),
            OutlierRemovalStep(contamination=0.03),  # filtra outliers
            NormalizationStep('standard'),  # escala ajuda critério
            # sem PCA por enquanto
            FeatureSelectionStep(method='mutual_info', percentile=95)  # corta só 5 %
        ]

        # Balanceamento antes das transformações ↓
        self.balancing_strategy = SMOTEBalancing(
            method='smote_enn',  # melhor que smote puro
            sampling_strategy='not majority'
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()

