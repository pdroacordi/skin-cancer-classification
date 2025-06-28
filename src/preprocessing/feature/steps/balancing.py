from abc import abstractmethod, ABC
from typing import Tuple, Dict, Any

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, KMeansSMOTE, ADASYN

from preprocessing.feature.base.balancing import BalancingStrategy


class SMOTEBalancing(BalancingStrategy):
    """SMOTE-based balancing strategies."""

    def __init__(self, method: str = 'smote', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.balancer = None

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.method == 'smote':
            self.balancer = SMOTE(random_state=42, **self.kwargs)
        elif self.method == 'kmeans_smote':
            self.balancer = KMeansSMOTE(random_state=42, **self.kwargs)
        elif self.method == 'adasyn':
            self.balancer = ADASYN(random_state=42, **self.kwargs)
        elif self.method == 'smote_enn':
            self.balancer = SMOTEENN(random_state=42, **self.kwargs)

        return self.balancer.fit_resample(X, y)

    def get_params(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'parameters': self.kwargs
        }


class TargetedSMOTEBalancing(SMOTEBalancing):
    """SMOTE that only oversamples very minority classes."""

    def __init__(self, minority_threshold=500, **kwargs):
        super().__init__(method='smote', **kwargs)
        self.minority_threshold = minority_threshold

    def balance(self, X: np.ndarray, y: np.ndarray):
        from imblearn.over_sampling import SMOTE
        from collections import Counter

        # Count samples per class
        class_counts = Counter(y)

        # Determine which classes to oversample
        sampling_strategy = {}
        max_samples = max(class_counts.values())

        for class_label, count in class_counts.items():
            if count < self.minority_threshold:
                # For very minority classes, increase significantly but not to max
                # This prevents creating too many synthetic samples
                target_count = min(count * 3, max_samples)
                sampling_strategy[class_label] = target_count
            else:
                # Keep original count for other classes
                sampling_strategy[class_label] = count

        # Apply SMOTE with custom strategy
        self.balancer = SMOTE(
            sampling_strategy=sampling_strategy,
            **self.kwargs
        )

        return self.balancer.fit_resample(X, y)


class ClassWeightBalancing(BalancingStrategy):
    """Class weight-based balancing."""

    def __init__(self):
        self.class_weights = None

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))
        # Don't actually resample, just compute weights
        return X, y

    def get_params(self) -> Dict[str, Any]:
        return {
            'class_weights': self.class_weights
        }
