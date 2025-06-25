"""
Feature selection steps following the same pattern as graphic/steps/
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, RFE, SelectFromModel,
    f_classif, mutual_info_classif, chi2
)
from sklearn.linear_model import LassoCV

from ..base.preprocessor import SupervisedFeaturePreprocessor, FeatureTransformationMixin


class FeatureSelector(SupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Base feature selector with various selection methods."""

    def __init__(self, method: str = 'mutual_info', k: Optional[int] = None,
                 percentile: Optional[float] = None, **kwargs):
        """
        Initialize feature selector.

        Args:
            method: Selection method ('f_score', 'mutual_info', 'chi2', 'rfe', 'lasso')
            k: Number of features to select
            percentile: Percentile of features to select (alternative to k)
            **kwargs: Additional arguments for the specific selector
        """
        self.method = method
        self.k = k
        self.percentile = percentile
        self.selector_kwargs = kwargs
        self.selector = None
        self.selected_features_ = None
        self.feature_scores_ = None

        # Validate method
        valid_methods = ['f_score', 'mutual_info', 'chi2', 'rfe', 'lasso', 'random_forest']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        if k is not None and percentile is not None:
            raise ValueError("Cannot specify both k and percentile")

    def _fit(self, features: np.ndarray, labels: np.ndarray) -> 'FeatureSelector':
        """Fit the feature selector."""
        self._validate_features(features)
        self._validate_labels(labels, features.shape[0])

        features = self._ensure_float32(features)
        n_features = features.shape[1]

        # Determine number of features to select
        if self.k is not None:
            k = min(self.k, n_features)
        elif self.percentile is not None:
            k = max(1, int(n_features * self.percentile / 100))
        else:
            # Auto-determine k
            k = min(max(10, n_features // 4), n_features)

        # Create appropriate selector
        if self.method == 'f_score':
            self.selector = SelectKBest(score_func=f_classif, k=k, **self.selector_kwargs)

        elif self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k, **self.selector_kwargs)

        elif self.method == 'chi2':
            # Ensure non-negative features for chi2
            if np.any(features < 0):
                print("Warning: Chi2 requires non-negative features, shifting to positive range")
                features = features - np.min(features, axis=0) + 1e-8
            self.selector = SelectKBest(score_func=chi2, k=k, **self.selector_kwargs)

        elif self.method == 'rfe':
            estimator = self.selector_kwargs.get('estimator',
                                                 RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
            self.selector = RFE(estimator=estimator, n_features_to_select=k,
                                **{k: v for k, v in self.selector_kwargs.items() if k != 'estimator'})

        elif self.method == 'lasso':
            # Use LassoCV for automatic regularization
            lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, **self.selector_kwargs)
            lasso.fit(features, labels)

            # Select features with non-zero coefficients
            selected_mask = np.abs(lasso.coef_) > 1e-5
            if np.sum(selected_mask) == 0:
                # If no features selected, select top k based on coefficient magnitude
                selected_mask = np.argsort(np.abs(lasso.coef_))[-k:] if k <= len(lasso.coef_) else np.ones(
                    len(lasso.coef_), dtype=bool)

            self.selected_features_ = selected_mask
            self.feature_scores_ = np.abs(lasso.coef_)
            return self

        elif self.method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, **self.selector_kwargs)
            self.selector = SelectFromModel(rf, max_features=k, threshold=-np.inf)

        # Fit the selector
        self.selector.fit(features, labels)

        # Store selected features and scores
        if hasattr(self.selector, 'get_support'):
            self.selected_features_ = self.selector.get_support()

        if hasattr(self.selector, 'scores_'):
            self.feature_scores_ = self.selector.scores_
        elif hasattr(self.selector, 'ranking_'):
            # For RFE, convert ranking to scores (lower rank = higher score)
            self.feature_scores_ = 1.0 / (self.selector.ranking_ + 1e-8)
        elif hasattr(self.selector, 'estimator_') and hasattr(self.selector.estimator_, 'feature_importances_'):
            self.feature_scores_ = self.selector.estimator_.feature_importances_

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        self._validate_features(features)
        features = self._ensure_float32(features)

        if self.method == 'lasso' and self.selected_features_ is not None:
            return features[:, self.selected_features_]
        elif self.selector is not None:
            return self._ensure_float32(self.selector.transform(features))
        else:
            return features

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Feature Selection ({self.method})"

    def get_params(self) -> dict:
        """Get selector parameters."""
        params = {
            'method': self.method,
            'k': self.k,
            'percentile': self.percentile,
            'n_selected': np.sum(self.selected_features_) if self.selected_features_ is not None else None
        }
        if self.selector is not None:
            params.update(self.selector.get_params())
        return params

    def get_selected_features(self) -> Optional[np.ndarray]:
        """Get mask of selected features."""
        return self.selected_features_

    def get_feature_scores(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.feature_scores_

    def get_feature_ranking(self) -> Optional[np.ndarray]:
        """Get feature ranking (1 = best)."""
        if self.feature_scores_ is not None:
            return np.argsort(np.argsort(self.feature_scores_)[::-1]) + 1
        return None


class MutualInfoFeatureSelector(FeatureSelector):
    """Feature selector using mutual information."""

    def __init__(self, k: Optional[int] = None, percentile: Optional[float] = None,
                 discrete_features: str = 'auto', random_state: int = 42):
        """
        Initialize mutual information selector.

        Args:
            k: Number of features to select
            percentile: Percentile of features to select
            discrete_features: How to treat discrete features
            random_state: Random state for reproducibility
        """
        super().__init__(
            method='mutual_info',
            k=k,
            percentile=percentile,
            discrete_features=discrete_features,
            random_state=random_state
        )

    def get_name(self) -> str:
        return "Mutual Information Feature Selection"


class StatisticalFeatureSelector(FeatureSelector):
    """Feature selector using statistical tests (F-score)."""

    def __init__(self, k: Optional[int] = None, percentile: Optional[float] = None):
        """
        Initialize statistical feature selector.

        Args:
            k: Number of features to select
            percentile: Percentile of features to select
        """
        super().__init__(method='f_score', k=k, percentile=percentile)

    def get_name(self) -> str:
        return "Statistical Feature Selection (F-score)"


class RFEFeatureSelector(FeatureSelector):
    """Recursive feature elimination selector."""

    def __init__(self, k: Optional[int] = None, estimator=None, step: int = 1):
        """
        Initialize RFE selector.

        Args:
            k: Number of features to select
            estimator: Base estimator for feature ranking
            step: Number of features to remove at each iteration
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

        super().__init__(method='rfe', k=k, estimator=estimator, step=step)

    def get_name(self) -> str:
        return "Recursive Feature Elimination"


class LassoFeatureSelector(FeatureSelector):
    """LASSO-based feature selector."""

    def __init__(self, alpha=None, cv: int = 5, random_state: int = 42):
        """
        Initialize LASSO selector.

        Args:
            alpha: Regularization strength (auto-selected if None)
            cv: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        kwargs = {'cv': cv, 'random_state': random_state}
        if alpha is not None:
            kwargs['alphas'] = [alpha]

        super().__init__(method='lasso', **kwargs)

    def get_name(self) -> str:
        return "LASSO Feature Selection"


class AdaptiveFeatureSelector(SupervisedFeaturePreprocessor, FeatureTransformationMixin):
    """Adaptive feature selector that combines multiple selection methods."""

    def __init__(self, methods: list = None, combination_strategy: str = 'intersection',
                 min_votes: int = 2, k: Optional[int] = None):
        """
        Initialize adaptive selector.

        Args:
            methods: List of selection methods to combine
            combination_strategy: How to combine selections ('intersection', 'union', 'voting')
            min_votes: Minimum votes needed for 'voting' strategy
            k: Target number of features
        """
        if methods is None:
            methods = ['mutual_info', 'f_score', 'rfe']

        self.methods = methods
        self.combination_strategy = combination_strategy
        self.min_votes = min_votes
        self.k = k
        self.selectors = {}
        self.feature_votes = None
        self.selected_features_ = None
        self.method_selections = {}

    def _fit(self, features: np.ndarray, labels: np.ndarray) -> 'AdaptiveFeatureSelector':
        """Fit multiple selectors and combine their selections."""
        self._validate_features(features)
        self._validate_labels(labels, features.shape[0])

        features = self._ensure_float32(features)
        n_features = features.shape[1]

        # Determine k for each method
        method_k = self.k if self.k is not None else max(10, n_features // 4)

        # Initialize vote counting
        self.feature_votes = np.zeros(n_features)

        # Fit each selector
        for method in self.methods:
            try:
                selector = FeatureSelector(method=method, k=method_k)
                selector.fit(features, labels)

                selected = selector.get_selected_features()
                if selected is not None:
                    self.selectors[method] = selector
                    self.method_selections[method] = selected
                    self.feature_votes += selected.astype(int)

            except Exception as e:
                print(f"Warning: Method {method} failed: {e}")
                continue

        if not self.selectors:
            raise ValueError("All feature selection methods failed")

        # Combine selections based on strategy
        if self.combination_strategy == 'intersection':
            # Features selected by all methods
            self.selected_features_ = self.feature_votes == len(self.selectors)

        elif self.combination_strategy == 'union':
            # Features selected by any method
            self.selected_features_ = self.feature_votes > 0

        elif self.combination_strategy == 'voting':
            # Features selected by at least min_votes methods
            self.selected_features_ = self.feature_votes >= self.min_votes

        else:
            raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")

        # Ensure we have at least some features selected
        if np.sum(self.selected_features_) == 0:
            print("Warning: No features selected by combination strategy, using top voted features")
            if self.k is not None:
                top_indices = np.argsort(self.feature_votes)[-self.k:]
            else:
                top_indices = np.argsort(self.feature_votes)[-max(1, n_features // 10):]
            self.selected_features_ = np.zeros(n_features, dtype=bool)
            self.selected_features_[top_indices] = True

        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using combined selection."""
        self._validate_features(features)

        if self.selected_features_ is None:
            raise ValueError("AdaptiveFeatureSelector must be fitted first")

        return self._ensure_float32(features[:, self.selected_features_])

    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        return f"Adaptive Feature Selection ({self.combination_strategy})"

    def get_params(self) -> dict:
        """Get adaptive selector parameters."""
        return {
            'methods': self.methods,
            'combination_strategy': self.combination_strategy,
            'min_votes': self.min_votes,
            'k': self.k,
            'n_selected': np.sum(self.selected_features_) if self.selected_features_ is not None else None,
            'successful_methods': list(self.selectors.keys())
        }

    def get_selection_report(self) -> dict:
        """Get detailed report about feature selection."""
        if self.feature_votes is None:
            return {}

        return {
            'total_features': len(self.feature_votes),
            'selected_features': np.sum(self.selected_features_) if self.selected_features_ is not None else 0,
            'feature_votes': self.feature_votes,
            'method_selections': self.method_selections,
            'vote_distribution': {
                f"{i}_votes": np.sum(self.feature_votes == i)
                for i in range(len(self.selectors) + 1)
            },
            'consensus_features': np.sum(self.feature_votes == len(self.selectors)),
            'controversial_features': np.sum((self.feature_votes > 0) & (self.feature_votes < len(self.selectors)))
        }

    def get_selected_features(self) -> Optional[np.ndarray]:
        """Get mask of selected features."""
        return self.selected_features_

    def get_feature_votes(self) -> Optional[np.ndarray]:
        """Get vote count for each feature."""
        return self.feature_votes