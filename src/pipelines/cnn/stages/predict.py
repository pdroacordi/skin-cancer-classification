"""
CNN inference stage.

Single responsibility: given (model, paths, labels, inference_config), produce
(y_true, y_pred, y_prob, uncertainty).  No metric computation, no file writes.

The old evaluate_model() mixed inference + metrics + I/O in a single 160-line
function.  It also stored MC Dropout uncertainty as a function attribute
(evaluate_model._uncertainty), a stateful anti-pattern that caused silent
bugs when called more than once in the same process.

This module fixes both problems: inference is pure computation, uncertainty
is returned explicitly as a fourth value.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from utils.data_loaders import MemoryEfficientDataGenerator


def predict_batch(
    model,
    paths: np.ndarray,
    labels: np.ndarray,
    cnn_model: str,
    batch_size: int,
    augment_fn: Optional[Callable] = None,
    use_mc_dropout: bool = False,
    mc_dropout_steps: int = 50,
    use_tta: bool = False,
    tta_n_steps: int = 8,
    tta_augment_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Run inference on a set of images.

    Supports three inference modes (mutually exclusive, in priority order):
    1. MC Dropout  (use_mc_dropout=True)
    2. TTA         (use_tta=True)
    3. Standard    (default)

    Args:
        model:            Trained Keras model.
        paths:            Image file paths.
        labels:           Integer ground-truth labels.
        cnn_model:        Backbone name for model-specific preprocessing.
        batch_size:       Images per forward pass.
        augment_fn:       Optional base augmentation (unused for MC/TTA — each
                          mode supplies its own augmentation).
        use_mc_dropout:   Enable Monte Carlo Dropout (Gal & Ghahramani, 2016).
        mc_dropout_steps: Number of stochastic forward passes per sample.
        use_tta:          Enable Test-Time Augmentation.
        tta_n_steps:      Number of augmented copies to average per sample.
        tta_augment_fn:   Augmentation function for TTA copies.

    Returns:
        y_true:      (N,) integer ground-truth labels.
        y_pred:      (N,) integer predicted labels.
        y_prob:      (N, C) softmax class probabilities.
        uncertainty: (N,) per-sample predictive variance when use_mc_dropout
                     is True; None otherwise.
    """
    gen = MemoryEfficientDataGenerator(
        paths=paths,
        labels=labels,
        batch_size=batch_size,
        model_name=cnn_model,
        augment_fn=augment_fn,
        shuffle=False,
    )
    n_steps = math.ceil(len(paths) / batch_size)

    y_true_list:  list = []
    y_pred_list:  list = []
    y_prob_list:  list = []
    uncertainty_list: Optional[list] = [] if use_mc_dropout else None

    for step_idx in range(n_steps):
        try:
            X_batch, y_batch = next(gen)
        except StopIteration:
            break

        if use_mc_dropout:
            prob_batch, step_uncertainty = _mc_dropout_pass(
                model, X_batch, mc_dropout_steps
            )
            uncertainty_list.extend(step_uncertainty.tolist())

        elif use_tta:
            prob_batch = _tta_pass(
                model, X_batch, paths, labels, step_idx,
                batch_size, cnn_model, tta_n_steps, tta_augment_fn,
            )

        else:
            prob_batch = model.predict(X_batch, verbose=0)

        y_true_list.extend(np.argmax(y_batch, axis=1))
        y_pred_list.extend(np.argmax(prob_batch, axis=1))
        y_prob_list.extend(prob_batch)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)
    uncertainty = np.array(uncertainty_list) if uncertainty_list is not None else None

    return y_true, y_pred, y_prob, uncertainty


# ------------------------------------------------------------------ #
#  Private helpers                                                     #
# ------------------------------------------------------------------ #

def _mc_dropout_pass(
    model,
    X_batch: np.ndarray,
    mc_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run *mc_steps* stochastic forward passes and return the mean
    prediction and per-sample predictive variance.

    Variance is computed incrementally (Welford-style accumulation)
    to avoid allocating mc_steps full tensors simultaneously.
    Gal & Ghahramani (2016, ICML, "Dropout as a Bayesian Approximation").
    """
    # First pass — also determines output shape for accumulators.
    first_sample = model(X_batch, training=True).numpy()
    mc_sum    = first_sample.copy()
    mc_sq_sum = first_sample ** 2

    for _ in range(mc_steps - 1):
        sample = model(X_batch, training=True).numpy()
        mc_sum    += sample
        mc_sq_sum += sample ** 2

    mean_prob = mc_sum / mc_steps
    # Mean variance over class dimension → scalar uncertainty per sample
    per_sample_variance = (mc_sq_sum / mc_steps - mean_prob ** 2).mean(axis=1)

    return mean_prob, per_sample_variance


def _tta_pass(
    model,
    X_batch: np.ndarray,
    all_paths: np.ndarray,
    all_labels: np.ndarray,
    step_idx: int,
    batch_size: int,
    cnn_model: str,
    tta_n_steps: int,
    tta_augment_fn: Optional[Callable],
) -> np.ndarray:
    """
    Average predictions over TTA_N_STEPS augmented copies of the current batch.
    The first copy is the original (no augmentation), the rest are augmented.

    Re-reads images from disk for each augmented copy because X_batch has
    already been preprocessed (backbone-specific normalization) and cannot
    be augmented in-place.
    """
    batch_paths  = all_paths [step_idx * batch_size: (step_idx + 1) * batch_size]
    batch_labels = all_labels[step_idx * batch_size: (step_idx + 1) * batch_size]

    preds = [model.predict(X_batch, verbose=0)]

    for _ in range(tta_n_steps - 1):
        tta_gen = MemoryEfficientDataGenerator(
            paths=batch_paths,
            labels=batch_labels,
            batch_size=len(batch_paths),
            model_name=cnn_model,
            augment_fn=tta_augment_fn,
            shuffle=False,
        )
        try:
            X_aug, _ = next(tta_gen)
            preds.append(model.predict(X_aug, verbose=0))
        except StopIteration:
            break

    return np.mean(preds, axis=0)
