"""
Dermoscopy-specific color normalization using the Reinhard method.

Normalizes each image to match a reference LAB color distribution computed
from the training set, reducing device- and acquisition-specific color biases.

Reference:
  Reinhard et al. (2001, IEEE CGA, "Color Transfer between Images").
  Applied to medical imaging: analogous to stain normalization in histopathology
  (Vahadane et al., 2016, IEEE TMI); novel for dermoscopy on HAM10000.

Usage:
  # Offline: compute reference statistics from training images
  stats = compute_reference_stats(train_image_paths)
  joblib.dump(stats, 'res/color_norm_stats.joblib')

  # Per-image normalization
  normalized = reinhard_normalize(img_bgr, stats['mean'], stats['std'])
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, List


def _to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to float64 LAB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    """Convert float64 LAB image back to BGR uint8."""
    clipped = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(clipped, cv2.COLOR_LAB2BGR)


def compute_reference_stats(image_paths: List[str]) -> dict:
    """
    Compute per-channel mean and std of the LAB representation for a set of
    training images. These statistics are used as the normalization target.

    Args:
        image_paths: List of paths to BGR JPEG/PNG images (training split only).

    Returns:
        dict with keys 'mean' and 'std', each a numpy array of shape (3,)
        representing L, A, B channel statistics.
    """
    all_means = []
    all_stds = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        lab = _to_lab(img)
        all_means.append(lab.mean(axis=(0, 1)))
        all_stds.append(lab.std(axis=(0, 1)))

    if not all_means:
        raise ValueError("No valid images found in image_paths")

    return {
        'mean': np.mean(all_means, axis=0),
        'std':  np.mean(all_stds,  axis=0),
    }


def reinhard_normalize(
        img_bgr: np.ndarray,
        reference_mean: np.ndarray,
        reference_std: np.ndarray
) -> np.ndarray:
    """
    Normalize a BGR image to match a reference LAB color distribution.

    Each LAB channel of the source image is z-scored and then re-scaled to
    the reference mean and std. This is a per-image global operation (~5 ms).

    Args:
        img_bgr: Source image as BGR uint8 array of shape (H, W, 3).
        reference_mean: Target per-channel LAB mean, shape (3,).
        reference_std:  Target per-channel LAB std, shape (3,).

    Returns:
        Normalized BGR uint8 array of the same shape as img_bgr.
    """
    source_lab = _to_lab(img_bgr)
    source_mean = source_lab.mean(axis=(0, 1))
    source_std  = source_lab.std(axis=(0, 1)) + 1e-6  # avoid div-by-zero

    normalized = (source_lab - source_mean) / source_std * reference_std + reference_mean
    return _from_lab(normalized)


class ColorNormalizationStep:
    """
    Preprocessing step that applies Reinhard color normalization.

    Fits on training images (saves mean/std via joblib), then normalizes
    all images at inference time. Integrates with PreprocessingPipeline.
    """

    def __init__(self, stats_path: str | None = None):
        """
        Args:
            stats_path: Path to joblib file containing {'mean': ..., 'std': ...}.
                        If None, the step is a no-op until fit() is called.
        """
        self.reference_mean: np.ndarray | None = None
        self.reference_std:  np.ndarray | None = None
        self.is_fitted = False

        if stats_path is not None:
            self.load(stats_path)

    def fit(self, image_paths: List[str], save_path: str | None = None):
        """
        Compute reference statistics from training images.

        Args:
            image_paths: Training image paths (never include val/test).
            save_path: Optional path to persist the stats with joblib.
        """
        stats = compute_reference_stats(image_paths)
        self.reference_mean = stats['mean']
        self.reference_std  = stats['std']
        self.is_fitted = True

        if save_path:
            import joblib
            joblib.dump(stats, save_path)
            print(f"Color normalization stats saved to: {save_path}")

    def load(self, stats_path: str):
        """Load previously computed statistics."""
        import joblib
        stats = joblib.load(stats_path)
        self.reference_mean = stats['mean']
        self.reference_std  = stats['std']
        self.is_fitted = True

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """Apply color normalization (no-op if not fitted)."""
        if not self.is_fitted:
            return img_bgr
        return reinhard_normalize(img_bgr, self.reference_mean, self.reference_std)

    def get_name(self) -> str:
        return "ReinhardColorNormalization"
