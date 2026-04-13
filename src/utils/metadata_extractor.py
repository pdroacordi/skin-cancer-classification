"""
Metadata feature extraction for HAM10000 dataset.
Extracts and encodes clinical metadata (age, sex, localization)
to be combined with CNN features.

Note: dx_type is intentionally excluded. It encodes how the ground-truth
diagnosis was confirmed (biopsy, consensus, follow-up), which is unavailable
at clinical prediction time and constitutes target leakage.
"""

import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Age bin boundaries (years)
# Boundaries follow standard clinical age groupings used in dermatology
# risk stratification (AAD guidelines, Siegel et al. 2022).
# ---------------------------------------------------------------------------
_AGE_BINS = [18, 30, 50, 65]   # < 18 | 18-29 | 30-49 | 50-64 | >= 65
_AGE_BIN_LABELS = ['pediatric', 'young_adult', 'adult', 'middle_age', 'senior']
_N_AGE_BINS = len(_AGE_BIN_LABELS)   # 5

# High-risk thresholds: very young (< 20) and older adults (> 60) have
# significantly elevated melanoma risk compared to the 20-60 age group
# (Hollestein & Nijsten 2014; Siegel et al. 2022).
_HIGH_RISK_AGE_YOUNG = 20
_HIGH_RISK_AGE_OLD = 60

# Sun-exposed localizations associated with UV-induced carcinogenesis.
_SUN_EXPOSED_SITES = frozenset({
    'face', 'neck', 'hand', 'arm', 'forearm',
    'ear', 'scalp', 'chest', 'back',
})

# Acral and palmoplantar sites — relevant for acral lentiginous melanoma subtype.
_ACRAL_SITES = frozenset({'hand', 'foot', 'toe', 'finger', 'palm', 'sole'})

# Trunk sites — distinct UV-exposure pattern from extremities.
_TRUNK_SITES = frozenset({'trunk', 'chest', 'back', 'abdomen'})


class MetadataFeatureExtractor:
    """
    Extract and encode clinical metadata features from HAM10000 dataset.

    Features extracted:
    - Age (StandardScaler normalized + 5 age-bin dummies + squared term)
    - Sex (one-hot encoded)
    - Localization (one-hot encoded)
    - Engineered: high_risk_age, sun_exposed_area, extremities, trunk_area
    """

    def __init__(self):
        self.age_scaler = StandardScaler()
        self.sex_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.localization_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        self.is_fitted = False
        self.feature_names: List[str] = []
        self.feature_indices: dict = {}

    def fit(self, metadata_df: pd.DataFrame) -> None:
        """
        Fit the encoders on the metadata.

        Args:
            metadata_df: DataFrame containing the metadata (train+val only —
                         never the full dataset, to prevent scaler leakage).
        """
        metadata_df = self._handle_missing_values(metadata_df)

        self.age_scaler.fit(metadata_df['age'].values.reshape(-1, 1))
        self.sex_encoder.fit(metadata_df[['sex']])
        self.localization_encoder.fit(metadata_df[['localization']])

        self._build_feature_names()
        self.is_fitted = True

    def transform(self, metadata_df: pd.DataFrame) -> np.ndarray:
        """
        Transform metadata into a fixed-length feature vector per sample.

        Args:
            metadata_df: DataFrame containing the metadata.

        Returns:
            Float32 feature matrix of shape (n_samples, n_features).
        """
        if not self.is_fitted:
            raise ValueError("MetadataFeatureExtractor must be fitted first")

        metadata_df = self._handle_missing_values(metadata_df)
        ages = metadata_df['age'].values

        age_features = self._extract_age_features(ages)
        sex_features = self.sex_encoder.transform(metadata_df[['sex']])
        loc_features = self.localization_encoder.transform(metadata_df[['localization']])
        eng_features = self._extract_engineered_features(metadata_df)

        return np.hstack([age_features, sex_features, loc_features, eng_features]).astype(np.float32)

    def fit_transform(self, metadata_df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(metadata_df)
        return self.transform(metadata_df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(df['age'].median())
        for col in ['sex', 'localization']:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        return df

    def _extract_age_features(self, ages: np.ndarray) -> np.ndarray:
        """Return (N, 7) matrix: [age_norm, 5 bin dummies, age_norm²]."""
        ages_2d = ages.reshape(-1, 1)
        age_norm = self.age_scaler.transform(ages_2d)           # (N, 1)

        # np.digitize assigns each value to the bin index 0..4 defined by
        # _AGE_BINS as left-open boundaries, matching the clinical groupings.
        bin_indices = np.digitize(ages, _AGE_BINS)              # (N,) values 0..4
        age_bins = np.eye(_N_AGE_BINS)[bin_indices]             # (N, 5) one-hot

        age_squared = age_norm ** 2                             # (N, 1)

        return np.hstack([age_norm, age_bins, age_squared])     # (N, 7)

    def _extract_engineered_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, 4) matrix of binary risk indicator features."""
        high_risk = (
            (df['age'] < _HIGH_RISK_AGE_YOUNG) | (df['age'] > _HIGH_RISK_AGE_OLD)
        ).astype(float).values.reshape(-1, 1)

        sun_exposed = df['localization'].isin(_SUN_EXPOSED_SITES).astype(float).values.reshape(-1, 1)
        acral = df['localization'].isin(_ACRAL_SITES).astype(float).values.reshape(-1, 1)
        trunk = df['localization'].isin(_TRUNK_SITES).astype(float).values.reshape(-1, 1)

        return np.hstack([high_risk, sun_exposed, acral, trunk])

    def _build_feature_names(self) -> None:
        """Populate ``self.feature_names`` and ``self.feature_indices``."""
        names: List[str] = []

        names.extend(['age_normalized'])
        names.extend([f'age_{label}' for label in _AGE_BIN_LABELS])
        names.extend(['age_squared'])

        sex_cats = self.sex_encoder.categories_[0]
        names.extend([f'sex_{c}' for c in sex_cats])

        loc_cats = self.localization_encoder.categories_[0]
        names.extend([f'loc_{c}' for c in loc_cats])

        names.extend(['high_risk_age', 'sun_exposed_area', 'acral_site', 'trunk_area'])

        self.feature_names = names
        self.feature_indices = {name: i for i, name in enumerate(names)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Persist the fitted extractor via joblib."""
        joblib.dump({
            'age_scaler': self.age_scaler,
            'sex_encoder': self.sex_encoder,
            'localization_encoder': self.localization_encoder,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'feature_indices': self.feature_indices,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'MetadataFeatureExtractor':
        """Load a fitted extractor from *filepath*."""
        extractor = cls()
        data = joblib.load(filepath)
        extractor.age_scaler = data['age_scaler']
        extractor.sex_encoder = data['sex_encoder']
        extractor.localization_encoder = data['localization_encoder']
        extractor.is_fitted = data['is_fitted']
        extractor.feature_names = data['feature_names']
        extractor.feature_indices = data['feature_indices']
        return extractor


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def combine_cnn_and_metadata_features(
    cnn_features: np.ndarray,
    metadata_features: Optional[np.ndarray],
) -> np.ndarray:
    """
    L2-normalise then concatenate CNN and metadata feature matrices.

    Each block is normalised independently before concatenation so that the
    very different magnitude scales of CNN activations and binary metadata
    indicators do not dominate each other.

    Args:
        cnn_features:      (N, D_cnn) float32 CNN feature matrix.
        metadata_features: (N, D_meta) float32 metadata features, or ``None``
                           if ``USE_METADATA=False``.

    Returns:
        (N, D_cnn + D_meta) or (N, D_cnn) float32 combined matrix.

    Raises:
        ValueError: If the two matrices have different numbers of samples.
    """
    if metadata_features is None:
        return cnn_features.astype(np.float32)

    if cnn_features.shape[0] != metadata_features.shape[0]:
        raise ValueError(
            f"Shape mismatch: CNN features have {cnn_features.shape[0]} samples "
            f"but metadata features have {metadata_features.shape[0]} samples."
        )

    eps = 1e-8
    cnn_norm = cnn_features / (np.linalg.norm(cnn_features, axis=1, keepdims=True) + eps)
    meta_norm = metadata_features / (np.linalg.norm(metadata_features, axis=1, keepdims=True) + eps)

    return np.hstack([cnn_norm, meta_norm]).astype(np.float32)


def extract_metadata_for_paths(
    image_paths: List[str],
    metadata_df: pd.DataFrame,
    metadata_extractor: MetadataFeatureExtractor,
) -> np.ndarray:
    """
    Extract metadata features for a list of image paths.

    The returned rows are aligned with *image_paths* (same order).

    Args:
        image_paths:        Ordered list of image file paths.
        metadata_df:        Full HAM10000 metadata DataFrame.
        metadata_extractor: A fitted ``MetadataFeatureExtractor``.

    Returns:
        (N, n_features) float32 metadata feature matrix.
    """
    image_ids = [os.path.basename(p).replace('.jpg', '') for p in image_paths]

    mask = metadata_df['image_id'].isin(image_ids)
    subset = metadata_df[mask].copy()

    # Reorder rows to match the original image_paths order.
    subset['_sort_key'] = pd.Categorical(
        subset['image_id'], categories=image_ids, ordered=True
    )
    subset = subset.sort_values('_sort_key').drop(columns='_sort_key')

    return metadata_extractor.transform(subset)
