from typing import Any, Dict, Tuple

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, KMeansSMOTE

from preprocessing.feature.base.balancing import BalancingStrategy

# How aggressively to oversample very-minority classes in TargetedSMOTEBalancing.
# A factor of 3 was chosen empirically: it brings the smallest HAM10000 classes
# (df, vasc, ~100-200 samples) to a usable size without generating so many
# synthetic samples that the interpolation strays too far from the real manifold.
# The cap at max_samples prevents the minority class from exceeding the majority.
_MINORITY_OVERSAMPLE_FACTOR = 3


class SMOTEBalancing(BalancingStrategy):
    """SMOTE-based oversampling with selectable variant.

    Supported methods: ``'smote'``, ``'kmeans_smote'``, ``'adasyn'``,
    ``'smote_enn'`` (SMOTE + Edited Nearest Neighbours for hybrid cleaning).
    """

    def __init__(self, method: str = 'smote', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.balancer = None

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _method_map = {
            'smote':       lambda: SMOTE(random_state=42, **self.kwargs),
            'kmeans_smote': lambda: KMeansSMOTE(random_state=42, **self.kwargs),
            'adasyn':      lambda: ADASYN(random_state=42, **self.kwargs),
            'smote_enn':   lambda: SMOTEENN(random_state=42, **self.kwargs),
        }
        if self.method not in _method_map:
            raise ValueError(
                f"Unknown SMOTE method: '{self.method}'. "
                f"Choose from {list(_method_map)}."
            )
        self.balancer = _method_map[self.method]()
        return self.balancer.fit_resample(X, y)

    def get_params(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'parameters': self.kwargs,
        }


class TargetedSMOTEBalancing(SMOTEBalancing):
    """SMOTE that oversamples only the most underrepresented classes.

    Classes with fewer than *minority_threshold* samples are oversampled by
    ``_MINORITY_OVERSAMPLE_FACTOR`` (×3), capped at the majority-class count.
    Classes already above the threshold are left unchanged.  This prevents
    creating excessive synthetic samples for already-adequate classes.
    """

    def __init__(self, minority_threshold: int = 500, **kwargs):
        super().__init__(method='smote', **kwargs)
        self.minority_threshold = minority_threshold

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from collections import Counter
        class_counts = Counter(y)
        max_class_size = max(class_counts.values())

        sampling_strategy: Dict[int, int] = {}
        for class_label, count in class_counts.items():
            if count < self.minority_threshold:
                # Multiply by factor but never exceed the majority-class size.
                target = min(count * _MINORITY_OVERSAMPLE_FACTOR, max_class_size)
                sampling_strategy[class_label] = target
            else:
                # Keep unchanged so SMOTE does not touch already-adequate classes.
                sampling_strategy[class_label] = count

        self.balancer = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            **self.kwargs,
        )
        return self.balancer.fit_resample(X, y)


class ClassWeightBalancing(BalancingStrategy):
    """Compute per-class weights without resampling.

    Unlike SMOTE-based strategies this does not alter the training set.
    The computed weights are stored in ``self.class_weights`` and should be
    passed to the classifier's ``fit()`` via ``sample_weight``.

    Note: This class intentionally returns ``(X, y)`` unchanged.  The caller
    is responsible for reading ``class_weights`` and forwarding them to the
    classifier.
    """

    def __init__(self):
        self.class_weights: Dict[int, float] = {}

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes.tolist(), weights.tolist()))
        return X, y

    def get_params(self) -> Dict[str, Any]:
        return {'class_weights': self.class_weights}
