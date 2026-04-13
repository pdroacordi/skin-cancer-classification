"""
ConfigurablePreprocessingPipeline — builds a pipeline from an ALGORITHM_PIPELINE_CONFIGS entry.

This replaces the five near-identical algorithm files (adaboost.py, extratrees.py,
randomforest.py, svm.py, xgboost.py).  To change a pipeline's steps, edit
``configs.py`` only.
"""

from preprocessing.feature.base.algorithm import AlgorithmPreprocessingPipeline
from preprocessing.feature.algorithm.configs import ALGORITHM_PIPELINE_CONFIGS


# Map of step-name strings → (class, constructor-kwarg-names)
def _build_step(step_name: str, params: dict):
    """Instantiate the correct step class from a name + params dict."""
    if step_name == 'variance_threshold':
        from preprocessing.feature.steps.threshold import VarianceThresholdStep
        return VarianceThresholdStep(**params)

    if step_name == 'normalization':
        from preprocessing.feature.steps.normalization import NormalizationStep
        return NormalizationStep(**params)

    if step_name == 'outlier_removal':
        from preprocessing.feature.steps.outlier_detection import OutlierRemovalStep
        return OutlierRemovalStep(**params)

    if step_name == 'feature_selection':
        from preprocessing.feature.steps.selection import FeatureSelectionStep
        return FeatureSelectionStep(**params)

    if step_name == 'dimensionality_reduction':
        from preprocessing.feature.steps.dimensionality_reduction import DimensionalityReductionStep
        return DimensionalityReductionStep(**params)

    raise ValueError(
        f"Unknown preprocessing step '{step_name}'. "
        "Add it to _build_step() in configurable.py."
    )


def _build_balancing(strategy_name: str, params: dict):
    """Instantiate the correct balancing strategy from a name + params dict."""
    if strategy_name == 'class_weight':
        from preprocessing.feature.steps.balancing import ClassWeightBalancing
        return ClassWeightBalancing()

    if strategy_name == 'smote':
        from preprocessing.feature.steps.balancing import SMOTEBalancing
        return SMOTEBalancing(**params)

    raise ValueError(
        f"Unknown balancing strategy '{strategy_name}'. "
        "Add it to _build_balancing() in configurable.py."
    )


class ConfigurablePreprocessingPipeline(AlgorithmPreprocessingPipeline):
    """Preprocessing pipeline driven entirely by ALGORITHM_PIPELINE_CONFIGS.

    Usage::

        pipeline = ConfigurablePreprocessingPipeline('ExtraTrees')
        pipeline.fit(X_train, y_train)
        X_proc, y_proc = pipeline.transform(X_train, y_train, training=True)
    """

    def __init__(self, algorithm: str, random_state: int = 42):
        if algorithm not in ALGORITHM_PIPELINE_CONFIGS:
            raise ValueError(
                f"No configuration found for algorithm '{algorithm}'. "
                f"Available: {list(ALGORITHM_PIPELINE_CONFIGS)}"
            )
        self.algorithm = algorithm
        super().__init__(random_state)
        self._configure_pipeline()

    def _configure_pipeline(self):
        cfg = ALGORITHM_PIPELINE_CONFIGS[self.algorithm]

        for step_name, params in cfg['steps']:
            self.steps.append(_build_step(step_name, params))

        if cfg.get('balancing'):
            strategy_name, strategy_params = cfg['balancing']
            self.balancing_strategy = _build_balancing(strategy_name, strategy_params)
