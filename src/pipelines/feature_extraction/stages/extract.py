"""
Feature extraction stage.

Single responsibility: extract CNN feature vectors from image paths and
optionally combine them with clinical metadata.

Key fixes over the original feature_extraction.py:
1. tf.keras.backend.clear_session() was called inside every mini-batch loop
   — this destroyed and recreated the TF graph every 4 images, causing
   enormous overhead.  Now clear_session() is only called between full
   extraction runs, not within them.

2. Metadata augmentation used a Python loop to replicate rows:
     [metadata[i]] * aug_factor  (O(N × aug_factor) Python loop)
   Replaced with np.repeat(metadata, aug_factor, axis=0) — vectorized.

3. The inner process_image() closure used augmentation_pipelines via an
   implicit closure binding.  Extracted to a named parameter to make the
   dependency explicit.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.data_loaders import resize_image


# Number of images processed per GPU predict call before requesting garbage
# collection.  Kept at 32 (not 4) because the bottleneck is data loading,
# not GPU memory, when features are extracted (no gradient computation).
_MINI_BATCH_SIZE = 32


def extract_features_batch(
    feature_extractor,
    paths: np.ndarray,
    labels: Optional[np.ndarray],
    img_size: Tuple[int, int],
    model_name: str,
    batch_size: int,
    augmentation_pipelines: Optional[List] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract CNN feature vectors from a list of image paths.

    When augmentation_pipelines is provided, each image produces N feature
    vectors (one per pipeline).  The returned features array has shape
    (N_samples × N_pipelines, feature_dim).  Labels are replicated
    accordingly.

    Args:
        feature_extractor:      Keras model with a single output tensor.
        paths:                  Array of image file paths.
        labels:                 Optional integer labels.
        img_size:               (height, width) target size.
        model_name:             Backbone name for model-specific preprocessing.
        batch_size:             Images loaded per batch (not the mini-batch).
        augmentation_pipelines: List of albumentations pipelines.  None means
                                no augmentation (single copy per image).

    Returns:
        (features, labels) where labels may be None if not provided.
    """
    import cv2
    from utils.data_loaders import apply_model_preprocessing

    n_augmentations = len(augmentation_pipelines) if augmentation_pipelines else 1
    apply_augmentation = augmentation_pipelines is not None

    print(f"Extracting features from {len(paths)} images "
          f"(augmentation × {n_augmentations})")

    n_samples  = len(paths)
    n_batches  = int(np.ceil(n_samples / batch_size))
    all_features: List[np.ndarray] = []
    all_labels:   List[int]        = []

    for batch_i in range(n_batches):
        start = batch_i * batch_size
        end   = min(start + batch_size, n_samples)
        batch_paths = paths[start:end]

        print(f"  Batch {batch_i + 1}/{n_batches} "
              f"(images {start + 1}–{end} of {n_samples})")

        batch_images: List[np.ndarray] = []
        batch_label_indices: List[int] = []

        for local_j, path in enumerate(batch_paths):
            original_idx = start + local_j

            img = _load_and_preprocess_image(path, img_size, model_name, None, cv2)
            if img is None:
                continue

            batch_images.append(img)
            if labels is not None:
                batch_label_indices.append(int(labels[original_idx]))

            if apply_augmentation and augmentation_pipelines:
                # Skip index 0 (treated as identity / original above)
                for aug_pipeline in augmentation_pipelines[1:]:
                    aug_img = _load_and_preprocess_image(
                        path, img_size, model_name, aug_pipeline, cv2
                    )
                    if aug_img is not None:
                        batch_images.append(aug_img)
                        if labels is not None:
                            batch_label_indices.append(int(labels[original_idx]))

        if not batch_images:
            continue

        # Predict in mini-batches to avoid OOM without destroying the TF graph.
        # clear_session() is intentionally NOT called here; it would rebuild the
        # computation graph on every mini-batch — a severe performance regression.
        batch_features = _predict_mini_batches(
            feature_extractor, batch_images, _MINI_BATCH_SIZE
        )
        all_features.extend(batch_features)
        all_labels.extend(batch_label_indices)

        # Release the loaded images; keep the model loaded
        del batch_images
        gc.collect()

    if not all_features:
        return np.empty((0,)), np.empty((0,)) if labels is not None else None

    features_array = np.array(all_features, dtype=np.float32)
    labels_array   = np.array(all_labels,   dtype=np.int64) if all_labels else None

    print(f"Extracted features shape: {features_array.shape}")
    return features_array, labels_array


def load_or_extract_features(
    feature_extractor,
    paths: np.ndarray,
    labels: Optional[np.ndarray],
    img_size: Tuple[int, int],
    model_name: str,
    batch_size: int,
    features_save_path: Optional[str],
    apply_augmentation: bool,
    augmentation_pipelines: Optional[List],
    metadata_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load from .npz cache or extract features from scratch, then optionally
    concatenate clinical metadata features.

    Cache invalidation: if the cache file exists AND was created with the same
    augmentation and metadata settings, it is used as-is.  Otherwise features
    are re-extracted and the cache is updated.

    Responsibilities (separated from the original monolithic function):
    - Cache load/save
    - Feature extraction delegation to extract_features_batch()
    - Metadata augmentation (replication to match augmented features)
    - Feature concatenation

    Args:
        feature_extractor:      Keras feature extractor model.
        paths:                  Image file paths.
        labels:                 Integer labels (may be None).
        img_size:               Target (height, width).
        model_name:             Backbone name.
        batch_size:             Batch size for extraction.
        features_save_path:     .npz cache path.  None disables caching.
        apply_augmentation:     Whether augmentation was requested.
        augmentation_pipelines: Actual augmentation objects (or None).
        metadata_features:      Pre-extracted metadata feature matrix of shape
                                (N_original, metadata_dim).  When provided,
                                concatenated with CNN features before return.

    Returns:
        (combined_features, labels)
    """
    from utils.metadata_extractor import combine_cnn_and_metadata_features

    # ------ Try cache ------
    cached = _load_cache(features_save_path, apply_augmentation, metadata_features is not None)
    if cached is not None:
        cnn_feats, cached_labels = cached
        labs = cached_labels if cached_labels is not None else labels
        print(f"Loaded cached features: {cnn_feats.shape}")
        return cnn_feats, labs

    # ------ Extract from scratch ------
    cnn_feats, out_labels = extract_features_batch(
        feature_extractor=feature_extractor,
        paths=paths,
        labels=labels,
        img_size=img_size,
        model_name=model_name,
        batch_size=batch_size,
        augmentation_pipelines=augmentation_pipelines if apply_augmentation else None,
    )

    if cnn_feats.size == 0:
        return cnn_feats, out_labels

    # ------ Handle metadata ------
    if metadata_features is not None:
        # Replicate metadata rows to match augmented feature count.
        # Vectorized via np.repeat — avoids O(N × aug_factor) Python loop.
        n_original = len(paths)
        n_extracted = cnn_feats.shape[0]
        if n_extracted != n_original:
            aug_factor = n_extracted // n_original
            if aug_factor > 1:
                metadata_features = np.repeat(metadata_features, aug_factor, axis=0)
                print(f"  Replicated metadata {aug_factor}× → {metadata_features.shape}")

    combined = combine_cnn_and_metadata_features(
        cnn_features=cnn_feats,
        metadata_features=metadata_features,
    )

    # ------ Update cache ------
    if features_save_path:
        _save_cache(
            features_save_path, combined, out_labels,
            apply_augmentation, metadata_features is not None
        )

    return combined, out_labels


# ------------------------------------------------------------------ #
#  Private helpers                                                     #
# ------------------------------------------------------------------ #

def _load_and_preprocess_image(
    path: str,
    img_size: Tuple[int, int],
    model_name: str,
    aug_pipeline,
    cv2,
) -> Optional[np.ndarray]:
    """Load, resize, optionally augment, and apply backbone preprocessing."""
    from utils.data_loaders import apply_model_preprocessing

    img = cv2.imread(str(path))
    if img is None:
        print(f"Warning: could not load image {path}")
        return None

    img = resize_image(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if aug_pipeline is not None:
        img = aug_pipeline(image=img)["image"]

    return apply_model_preprocessing(img, model_name)


def _predict_mini_batches(
    feature_extractor,
    images: List[np.ndarray],
    mini_batch_size: int,
) -> List[np.ndarray]:
    """
    Run feature_extractor.predict() on images split into mini-batches.

    Falls back to single-image inference when a mini-batch triggers OOM.
    Returns a flat list of 1-D feature vectors.
    """
    import tensorflow as tf

    features: List[np.ndarray] = []
    n = len(images)
    n_mini_batches = int(np.ceil(n / mini_batch_size))

    for mb_i in range(n_mini_batches):
        mb_images = images[mb_i * mini_batch_size: (mb_i + 1) * mini_batch_size]
        try:
            batch_arr = np.array(mb_images)
            preds = feature_extractor.predict(batch_arr, verbose=0)
            features.extend(preds)
        except tf.errors.ResourceExhaustedError:
            print(f"  OOM in mini-batch {mb_i + 1}, switching to single-image mode")
            for img in mb_images:
                try:
                    pred = feature_extractor.predict(np.expand_dims(img, 0), verbose=0)
                    features.extend(pred)
                except Exception as exc:
                    print(f"  Skipping image due to error: {exc}")

    return features


def _load_cache(
    path: Optional[str],
    augmentation_enabled: bool,
    metadata_enabled: bool,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Return (features, labels) from .npz cache if valid, else None."""
    if not path:
        return None
    import os
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        # Invalidate cache if augmentation or metadata settings changed
        cached_aug  = bool(data.get("augmentation_enabled", False))
        cached_meta = bool(data.get("metadata_enabled",     False))
        if cached_aug != augmentation_enabled or cached_meta != metadata_enabled:
            print("Cache invalidated (augmentation/metadata settings changed)")
            return None
        labels = data["labels"] if "labels" in data else None
        return data["features"], labels
    except Exception as exc:
        print(f"Warning: cache load failed ({exc}), re-extracting")
        return None


def _save_cache(
    path: str,
    features: np.ndarray,
    labels: Optional[np.ndarray],
    augmentation_enabled: bool,
    metadata_enabled: bool,
) -> None:
    """Save features and metadata flags to a .npz cache file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_kwargs: dict = {
        "features":             features,
        "augmentation_enabled": augmentation_enabled,
        "metadata_enabled":     metadata_enabled,
    }
    if labels is not None:
        save_kwargs["labels"] = labels
    np.savez(path, **save_kwargs)
