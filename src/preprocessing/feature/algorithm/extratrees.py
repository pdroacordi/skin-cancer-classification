from imblearn.over_sampling import SMOTE

from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.steps.balancing import SMOTEBalancing, ClassWeightBalancing, TargetedSMOTEBalancing
from preprocessing.feature.steps.dimensionality_reduction import DimensionalityReductionStep
from preprocessing.feature.steps.normalization import NormalizationStep
from preprocessing.feature.steps.selection import FeatureSelectionStep
from preprocessing.feature.steps.threshold import VarianceThresholdStep


class ExtraTreesPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline optimized for ExtraTrees."""

    def _configure_pipeline(self):
        self.steps += [
            VarianceThresholdStep(0),
            NormalizationStep('robust'),
            FeatureSelectionStep(method='mutual_info', percentile=95)  # ou opção B
        ]

        # Balanceamento sem KMeans (evita erro de cluster)
        self.balancing_strategy = SMOTEBalancing(
            method='smote_enn',  # gera sintéticos + remove ruído
            sampling_strategy='not majority'
        )

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self._configure_pipeline()
