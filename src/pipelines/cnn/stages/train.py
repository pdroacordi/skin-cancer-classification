"""
CNN training stage — single fold.

Single responsibility: given (train_gen, val_gen, model_save_path), fit a CNN
and return (model, history).  No metric computation, no file writes,
no config globals.

All configuration values are passed explicitly as parameters so this
function is testable in isolation and free of import-time side effects.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import cfg
from models.cnn_models import (
    load_or_create_cnn, get_callbacks, compile_phase2, unfreeze_from,
)
from utils.data_loaders import MemoryEfficientDataGenerator


def train_one_fold(
    train_paths: np.ndarray,
    train_labels: np.ndarray,
    val_paths: np.ndarray,
    val_labels: np.ndarray,
    model_save_path: str,
    log_dir: str,
    cnn_model: str,
    batch_size: int,
    num_epochs: int,
    use_fine_tuning: bool,
    augment_fn: Optional[Callable] = None,
) -> Tuple[object, Optional[object]]:
    """
    Train a CNN classifier on one (train, val) fold.

    If a checkpoint already exists at model_save_path, loads and returns it
    without retraining (resume semantics).

    Args:
        train_paths:      Array of training image file paths.
        train_labels:     Integer class labels for training images.
        val_paths:        Array of validation image file paths.
        val_labels:       Integer class labels for validation images.
        model_save_path:  Where to save the best checkpoint (ModelCheckpoint).
        log_dir:          TensorBoard log directory.
        cnn_model:        Backbone name, e.g. "ResNet".
        batch_size:       Images per batch.
        num_epochs:       Maximum training epochs (early stopping may end sooner).
        use_fine_tuning:  Whether to unfreeze the upper CNN layers.
        augment_fn:       Optional albumentations transform applied per image.

    Returns:
        (model, history)  history is None when loading a pre-existing checkpoint.
    """
    train_gen = MemoryEfficientDataGenerator(
        paths=train_paths,
        labels=train_labels,
        batch_size=batch_size,
        model_name=cnn_model,
        augment_fn=augment_fn,
        shuffle=True,
    )
    val_gen = MemoryEfficientDataGenerator(
        paths=val_paths,
        labels=val_labels,
        batch_size=batch_size,
        model_name=cnn_model,
        augment_fn=None,
        shuffle=False,
    )

    model, already_loaded = load_or_create_cnn(
        model_name=cnn_model,
        mode="classifier",
        fine_tune=use_fine_tuning,
        save_path=model_save_path,
        n_train=len(train_paths),
    )

    if already_loaded:
        print(f"Loaded existing checkpoint: {model_save_path}")
        return model, None

    steps_per_epoch = math.ceil(len(train_paths) / batch_size)
    val_steps       = math.ceil(len(val_paths)   / batch_size)

    # Balanced class weights compensate for HAM10000's strong class imbalance
    # (~67 % of samples belong to 'nv').
    classes = np.unique(train_labels)
    class_weights = {
        int(c): float(w)
        for c, w in zip(
            classes,
            compute_class_weight("balanced", classes=classes, y=train_labels),
        )
    }

    callbacks = get_callbacks(model_save_path, log_dir)

    if use_fine_tuning:
        warmup = max(0, min(int(cfg.warmup_epochs), num_epochs - 1))
        # Phase 1 — head only.
        history1 = None
        if warmup > 0:
            print(f"Phase 1: head-only training for {warmup} warmup epochs")
            history1 = model.fit(
                train_gen.get_keras_generator(),
                steps_per_epoch=steps_per_epoch,
                epochs=warmup,
                validation_data=val_gen.get_keras_generator(),
                validation_steps=val_steps,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
            )

        # Phase 2 — unfreeze and keep training.
        remaining = num_epochs - warmup
        print(f"Phase 2: unfreezing from layer {cnn_model}[{cfg.fine_tune_from_layer}] "
              f"for {remaining} epochs")
        unfreeze_from(model, cnn_model)
        compile_phase2(
            model, n_train=len(train_paths),
            batch_size=batch_size, num_epochs=remaining,
        )
        history = model.fit(
            train_gen.get_keras_generator(),
            steps_per_epoch=steps_per_epoch,
            initial_epoch=warmup,
            epochs=num_epochs,
            validation_data=val_gen.get_keras_generator(),
            validation_steps=val_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )
    else:
        history = model.fit(
            train_gen.get_keras_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            validation_data=val_gen.get_keras_generator(),
            validation_steps=val_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

    return model, history
