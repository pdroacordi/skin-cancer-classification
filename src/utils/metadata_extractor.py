"""
Metadata feature extraction for HAM10000 dataset.
Extracts and encodes clinical metadata (age, sex, localization, dx_type)
to be combined with CNN features.
"""

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class MetadataFeatureExtractor:
    """
    Extract and encode clinical metadata features from HAM10000 dataset.

    Features extracted:
    - Age (normalized)
    - Sex (one-hot encoded)
    - Localization (one-hot encoded)
    - dx_type (one-hot encoded)
    """

    def __init__(self):
        self.age_scaler = StandardScaler()
        self.sex_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.localization_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.dx_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        self.is_fitted = False
        self.feature_names = []
        self.feature_indices = {}

    def fit(self, metadata_df: pd.DataFrame):
        """
        Fit the encoders on the metadata.

        Args:
            metadata_df: DataFrame containing the metadata
        """
        # Handle missing values
        metadata_df = self._handle_missing_values(metadata_df)

        # Fit age scaler
        ages = metadata_df['age'].values.reshape(-1, 1)
        self.age_scaler.fit(ages)

        # Fit categorical encoders
        self.sex_encoder.fit(metadata_df[['sex']])
        self.localization_encoder.fit(metadata_df[['localization']])
        self.dx_type_encoder.fit(metadata_df[['dx_type']])

        # Build feature names
        self._build_feature_names()

        self.is_fitted = True

    def transform(self, metadata_df: pd.DataFrame) -> np.ndarray:
        """
        Transform metadata into feature vectors.

        Args:
            metadata_df: DataFrame containing the metadata

        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("MetadataFeatureExtractor must be fitted first")

        # Handle missing values
        metadata_df = self._handle_missing_values(metadata_df)

        features = []

        # Age features (normalized + binned)
        age_features = self._extract_age_features(metadata_df['age'].values)
        features.append(age_features)

        # Sex features (one-hot)
        sex_features = self.sex_encoder.transform(metadata_df[['sex']])
        features.append(sex_features)

        # Localization features (one-hot)
        loc_features = self.localization_encoder.transform(metadata_df[['localization']])
        features.append(loc_features)

        # dx_type features (one-hot)
        dx_type_features = self.dx_type_encoder.transform(metadata_df[['dx_type']])
        features.append(dx_type_features)

        # Additional engineered features
        eng_features = self._extract_engineered_features(metadata_df)
        features.append(eng_features)

        # Concatenate all features
        return np.hstack(features).astype(np.float32)

    def fit_transform(self, metadata_df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(metadata_df)
        return self.transform(metadata_df)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the metadata."""
        df = df.copy()

        # Age: fill with median
        if 'age' in df.columns:
            median_age = df['age'].median()
            df['age'] = df['age'].fillna(median_age)

        # Categorical: fill with 'unknown'
        for col in ['sex', 'localization', 'dx_type']:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')

        return df

    def _extract_age_features(self, ages: np.ndarray) -> np.ndarray:
        """Extract age-related features."""
        ages = ages.reshape(-1, 1)

        # Normalized age
        age_normalized = self.age_scaler.transform(ages)

        # Age bins (pediatric, young adult, adult, middle age, senior)
        age_bins = np.zeros((len(ages), 5))
        age_bins[ages.flatten() < 18, 0] = 1  # pediatric
        age_bins[(ages.flatten() >= 18) & (ages.flatten() < 30), 1] = 1  # young adult
        age_bins[(ages.flatten() >= 30) & (ages.flatten() < 50), 2] = 1  # adult
        age_bins[(ages.flatten() >= 50) & (ages.flatten() < 65), 3] = 1  # middle age
        age_bins[ages.flatten() >= 65, 4] = 1  # senior

        # Age squared (for non-linear relationships)
        age_squared = (age_normalized ** 2)

        return np.hstack([age_normalized, age_bins, age_squared])

    def _extract_engineered_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract engineered features from metadata."""
        features = []

        # High-risk age groups (very young or elderly)
        high_risk_age = ((df['age'] < 20) | (df['age'] > 60)).astype(float).values.reshape(-1, 1)
        features.append(high_risk_age)

        # Sun-exposed areas (face, neck, hands, arms)
        sun_exposed = df['localization'].isin([
            'face', 'neck', 'hand', 'arm', 'forearm',
            'ear', 'scalp', 'chest', 'back'
        ]).astype(float).values.reshape(-1, 1)
        features.append(sun_exposed)

        # Extremities (hands, feet)
        extremities = df['localization'].isin([
            'hand', 'foot', 'toe', 'finger', 'palm', 'sole'
        ]).astype(float).values.reshape(-1, 1)
        features.append(extremities)

        # Trunk area
        trunk = df['localization'].isin([
            'trunk', 'chest', 'back', 'abdomen'
        ]).astype(float).values.reshape(-1, 1)
        features.append(trunk)

        # Clinical diagnosis confidence (histo > follow_up > consensus > single)
        dx_confidence = np.zeros((len(df), 1))
        dx_confidence[df['dx_type'] == 'histo'] = 1.0
        dx_confidence[df['dx_type'] == 'follow_up'] = 0.75
        dx_confidence[df['dx_type'] == 'consensus'] = 0.5
        dx_confidence[df['dx_type'] == 'confocal'] = 0.5
        features.append(dx_confidence)

        return np.hstack(features)

    def _build_feature_names(self):
        """Build list of feature names for interpretability."""
        self.feature_names = []

        # Age features
        self.feature_names.extend([
            'age_normalized',
            'age_pediatric', 'age_young_adult', 'age_adult',
            'age_middle_age', 'age_senior',
            'age_squared'
        ])

        # Sex features
        sex_categories = self.sex_encoder.categories_[0]
        self.feature_names.extend([f'sex_{cat}' for cat in sex_categories])

        # Localization features
        loc_categories = self.localization_encoder.categories_[0]
        self.feature_names.extend([f'loc_{cat}' for cat in loc_categories])

        # dx_type features
        dx_categories = self.dx_type_encoder.categories_[0]
        self.feature_names.extend([f'dx_type_{cat}' for cat in dx_categories])

        # Engineered features
        self.feature_names.extend([
            'high_risk_age',
            'sun_exposed_area',
            'extremities',
            'trunk_area',
            'dx_confidence'
        ])

        # Build feature indices for easy access
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}

    def get_feature_importance_mask(self, importance_threshold: float = 0.01) -> np.ndarray:
        """
        Get a mask for important features based on variance.

        Args:
            importance_threshold: Minimum variance threshold

        Returns:
            Boolean mask of important features
        """
        # This would be computed from actual feature importance
        # For now, return all features as important
        return np.ones(len(self.feature_names), dtype=bool)

    def save(self, filepath: str):
        """Save the fitted extractor."""
        save_data = {
            'age_scaler': self.age_scaler,
            'sex_encoder': self.sex_encoder,
            'localization_encoder': self.localization_encoder,
            'dx_type_encoder': self.dx_type_encoder,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'feature_indices': self.feature_indices
        }
        joblib.dump(save_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'MetadataFeatureExtractor':
        """Load a fitted extractor."""
        extractor = cls()
        save_data = joblib.load(filepath)

        extractor.age_scaler = save_data['age_scaler']
        extractor.sex_encoder = save_data['sex_encoder']
        extractor.localization_encoder = save_data['localization_encoder']
        extractor.dx_type_encoder = save_data['dx_type_encoder']
        extractor.is_fitted = save_data['is_fitted']
        extractor.feature_names = save_data['feature_names']
        extractor.feature_indices = save_data['feature_indices']

        return extractor


def combine_cnn_and_metadata_features(
        cnn_features: np.ndarray,
        metadata_features: np.ndarray
) -> np.ndarray:
    """
    Combine CNN features with metadata features.

    Args:
        cnn_features: CNN-extracted features (n_samples, n_cnn_features)
        metadata_features: Metadata features (n_samples, n_metadata_features)

    Returns:
        Combined feature matrix
    """
    if metadata_features is None:
        # No metadata, just return normalized CNN features
        return cnn_features.astype(np.float32)

        # Ensure same number of samples
    if cnn_features.shape[0] != metadata_features.shape[0]:
        raise ValueError(
            f"Shape mismatch: CNN features have {cnn_features.shape[0]} samples, "
            f"metadata features have {metadata_features.shape[0]} samples"
        )

        # Normalize features to same scale
    eps = 1e-8
    cnn_norm = cnn_features / (np.linalg.norm(cnn_features, axis=1, keepdims=True) + eps)
    meta_norm = metadata_features / (np.linalg.norm(metadata_features, axis=1, keepdims=True) + eps)

    # Concatenate
    return np.hstack([cnn_norm, meta_norm]).astype(np.float32)


def extract_metadata_for_paths(
        image_paths: List[str],
        metadata_df: pd.DataFrame,
        metadata_extractor: MetadataFeatureExtractor
) -> np.ndarray:
    """
    Extract metadata features for a list of image paths.

    Args:
        image_paths: List of image paths
        metadata_df: Full metadata DataFrame
        metadata_extractor: Fitted MetadataFeatureExtractor

    Returns:
        Metadata features for the given paths
    """

    # Extract image IDs from paths
    image_ids = []
    for path in image_paths:
        filename = os.path.basename(path)
        image_id = filename.replace('.jpg', '')
        image_ids.append(image_id)

    # Get metadata for these images
    mask = metadata_df['image_id'].isin(image_ids)
    subset_metadata = metadata_df[mask].copy()

    # Ensure same order as paths
    subset_metadata['image_id_for_sort'] = pd.Categorical(
        subset_metadata['image_id'],
        categories=image_ids,
        ordered=True
    )
    subset_metadata = subset_metadata.sort_values('image_id_for_sort')

    # Extract features
    return metadata_extractor.transform(subset_metadata)
