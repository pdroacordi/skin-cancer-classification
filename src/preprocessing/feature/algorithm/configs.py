"""
Declarative preprocessing pipeline configurations for each classical ML algorithm.

Each entry maps an algorithm name to a dict with:
  'steps'     — ordered list of (step_name, params_dict) tuples
  'balancing' — (strategy_name, params_dict) or None

Step names and their corresponding classes (resolved in configurable.py):
  'variance_threshold'      → VarianceThresholdStep(threshold=...)
  'normalization'           → NormalizationStep(method=...)
  'outlier_removal'         → OutlierRemovalStep(method=..., contamination=...)
  'feature_selection'       → FeatureSelectionStep(method=..., percentile=...)
  'dimensionality_reduction'→ DimensionalityReductionStep(method=..., variance_threshold=...)

Balancing strategy names:
  'class_weight' → ClassWeightBalancing()
  'smote'        → SMOTEBalancing(**params)

To add a new algorithm: add a new key here and register it in
``PreprocessingPipelineFactory._pipelines`` in ``pipeline.py``.
"""

ALGORITHM_PIPELINE_CONFIGS = {
    'ExtraTrees': {
        'steps': [
            ('variance_threshold', {'threshold': 0}),
            ('normalization',      {'method': 'robust'}),
            ('feature_selection',  {'method': 'mutual_info', 'percentile': 95}),
        ],
        'balancing': ('smote', {'method': 'smote_enn', 'sampling_strategy': 'not majority'}),
    },
    'RandomForest': {
        'steps': [
            ('variance_threshold', {'threshold': 0}),
            ('outlier_removal',    {'contamination': 0.03}),
            ('normalization',      {'method': 'standard'}),
            ('feature_selection',  {'method': 'mutual_info', 'percentile': 95}),
        ],
        'balancing': ('smote', {'method': 'smote_enn', 'sampling_strategy': 'not majority'}),
    },
    'XGBoost': {
        'steps': [
            ('variance_threshold',      {'threshold': 0}),
            ('outlier_removal',         {'contamination': 0.05}),
            ('normalization',           {'method': 'standard'}),
            ('dimensionality_reduction',{'method': 'pca', 'variance_threshold': 0.95}),
            ('feature_selection',       {'method': 'mutual_info', 'percentile': 90}),
        ],
        'balancing': ('smote', {'method': 'smote_enn', 'sampling_strategy': 'not majority'}),
    },
    # LightGBM: leaf-wise growth handles high-dimensional embeddings without PCA.
    # Class_weight balancing is preferred over SMOTE because SMOTE generates synthetic
    # points by interpolation in feature space — in CNN embedding space, interpolated
    # points do not correspond to valid skin lesion images and can mislead the classifier.
    'LightGBM': {
        'steps': [
            ('variance_threshold', {'threshold': 0}),
            ('normalization',      {'method': 'robust'}),
            ('feature_selection',  {'method': 'mutual_info', 'percentile': 90}),
        ],
        'balancing': ('class_weight', {}),
    },
    # HistGradientBoosting: sklearn-native GBDT; same pipeline as LightGBM for fair comparison.
    'HistGradientBoosting': {
        'steps': [
            ('variance_threshold', {'threshold': 0}),
            ('normalization',      {'method': 'robust'}),
            ('feature_selection',  {'method': 'mutual_info', 'percentile': 90}),
        ],
        'balancing': ('class_weight', {}),
    },
    'SVM': {
        'steps': [
            ('outlier_removal',         {'method': 'isolation_forest', 'contamination': 0.03}),
            ('variance_threshold',      {'threshold': 1e-6}),
            ('normalization',           {'method': 'standard'}),
            ('feature_selection',       {'method': 'mutual_info', 'percentile': 60}),
            ('dimensionality_reduction',{'method': 'pca', 'variance_threshold': 0.95}),
        ],
        'balancing': ('class_weight', {}),
    },
}
