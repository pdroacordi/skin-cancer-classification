"""
Centralised result directory naming for both pipelines.

Both pipelines previously built their result directory names from config flags
inline.  This module provides pure-string functions so that the naming
convention is defined in one place and can be tested independently.

Result directory conventions:
  CNN classifier:
    results/cnn_classifier_<MODEL>_[contrast_][hair_removal_][use_augmentation_]/

  Feature extraction:
    results/feature_extraction_<MODEL>_[contrast_][hair_removal_]
      [use_augmentation_][use_feature_augmentation_]
      [use_feature_preprocessing_][use_metadata_]/<classifier>/
"""

import os

from config import (
    CNN_MODEL,
    CLASSICAL_CLASSIFIER_MODEL,
    RESULTS_DIR,
    USE_HAIR_REMOVAL,
    USE_ENHANCED_CONTRAST,
    USE_GRAPHIC_PREPROCESSING,
    USE_DATA_AUGMENTATION,
    USE_FEATURE_AUGMENTATION,
    USE_FEATURE_PREPROCESSING,
    USE_METADATA,
)


def _graphic_suffix() -> str:
    """Return the preprocessing flag string common to both pipelines."""
    if not USE_GRAPHIC_PREPROCESSING:
        return ""
    contrast = "contrast_" if USE_ENHANCED_CONTRAST else ""
    hair     = "hair_removal_" if USE_HAIR_REMOVAL else ""
    return f"{contrast}{hair}"


def cnn_result_dir(base_dir: str = None) -> str:
    """Return the base result directory path for the CNN classifier pipeline."""
    base_dir = base_dir or RESULTS_DIR
    suffix = (
        _graphic_suffix()
        + ("use_augmentation_" if USE_DATA_AUGMENTATION else "")
    )
    return os.path.join(base_dir, f"cnn_classifier_{CNN_MODEL}_{suffix}")


def feature_extraction_experiment_dir(base_dir: str = None,
                                      cnn_model: str = None) -> str:
    """Return the experiment-level directory for the feature extraction pipeline.

    This is the directory that holds all classifiers for a given CNN + flag
    combination, e.g.::

        results/feature_extraction_ResNet_use_augmentation_/
    """
    base_dir  = base_dir  or RESULTS_DIR
    cnn_model = cnn_model or CNN_MODEL
    suffix = (
        _graphic_suffix()
        + ("use_augmentation_"          if USE_DATA_AUGMENTATION    else "")
        + ("use_feature_augmentation_"  if USE_FEATURE_AUGMENTATION  else "")
        + ("use_feature_preprocessing_" if USE_FEATURE_PREPROCESSING else "")
        + ("use_metadata_"              if USE_METADATA               else "")
    )
    return os.path.join(base_dir, f"feature_extraction_{cnn_model}_{suffix}")


def feature_extraction_result_dir(base_dir: str = None,
                                   cnn_model: str = None,
                                   classifier: str = None) -> str:
    """Return the classifier-level result directory for the feature extraction pipeline.

    Matches the layout documented in ``src/pipelines/CLAUDE.md``::

        results/feature_extraction_<MODEL>_<flags>/<classifier>/
    """
    classifier = classifier or CLASSICAL_CLASSIFIER_MODEL
    exp_dir    = feature_extraction_experiment_dir(base_dir, cnn_model)
    return os.path.join(exp_dir, classifier.lower())
