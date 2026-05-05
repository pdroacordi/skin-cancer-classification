"""
Experiment configuration for the skin cancer classification project.

All defaults live here. Override any value at runtime via the CLI:
    python main.py --cnn-model Xception --batch-size 32

Access from any module:
    from config import cfg
    model_name = cfg.cnn_model
    size       = cfg.img_size   # auto-derived from cfg.cnn_model

Mutation by apply_cli_overrides() is visible everywhere because cfg is a
shared mutable object — modules no longer need to defer their imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Backbone → canonical input resolution  (module-level, not mutable)
# ---------------------------------------------------------------------------
BACKBONE_IMG_SIZE: Dict[str, Tuple[int, int, int]] = {
    'Inception':       (299, 299, 3),
    'Xception':        (299, 299, 3),
    'ConvNeXt':        (224, 224, 3),
    'EfficientNetV2S': (384, 384, 3),
}

FINE_TUNING_AT_LAYER: Dict[str, int] = {
    'Inception':       280,
    'Xception':        100,
    'ConvNeXt':        60,     # last two stages of ConvNeXtTiny
    'EfficientNetV2S': 330,    # approximately last block start
}

VALID_BACKBONES:    List[str] = list(BACKBONE_IMG_SIZE)
VALID_CLASSIFIERS:  List[str] = [
    'RandomForest', 'XGBoost', 'LightGBM',
    'HistGradientBoosting', 'ExtraTrees', 'SVM',
]
VALID_DES_ALGORITHMS: List[str] = [
    'knorau', 'desmi', 'metades', 'singlebest',
    # H3 — ECE-guided DES
    'knorau-ece',
    'knorau-temp',
    # H16 — Conformal-Targeted DES
    'conformal-targeted',
    'conformal-targeted-ece',
    'conformal-targeted-random',
]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    All experiment hyperparameters and feature flags in one typed object.

    Derived values (img_size, data-file paths) are @property so they stay
    consistent whenever cnn_model or use_graphic_preprocessing is mutated.
    """

    # ---- Data ---------------------------------------------------------------
    num_classes: int = 7
    batch_size: int = 16
    metadata_path: str = "./res/metadata.csv"
    results_dir: str = "./results"

    # ---- Training -----------------------------------------------------------
    num_epochs: int = 100
    num_kfolds: int = 5
    num_iterations: int = 2
    num_final_models: int = 10

    # ---- Callbacks ----------------------------------------------------------
    early_stopping_patience: int = 10

    # ---- Optimizer / schedule ----------------------------------------------
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # ---- Model selection ----------------------------------------------------
    cnn_model: str = 'EfficientNetV2S'
    classical_classifier_model: str = 'ExtraTrees'

    # ---- Pipeline flags -----------------------------------------------------
    use_graphic_preprocessing: bool = False
    use_data_augmentation: bool = True
    use_feature_augmentation: bool = False
    use_feature_preprocessing: bool = True
    use_fine_tuning: bool = True
    use_metadata: bool = False
    use_hair_removal: bool = True
    use_enhanced_contrast: bool = False
    use_color_normalization: bool = False
    color_norm_stats_path: str = "./res/color_norm_stats.joblib"
    visualize: bool = False

    # ---- Loss ---------------------------------------------------------------
    use_focal_loss: bool = False
    label_smoothing: float = 0.0

    # ---- Inference ----------------------------------------------------------
    use_tta: bool = False
    tta_n_steps: int = 8
    use_mixup: bool = False
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    use_mc_dropout: bool = False
    mc_dropout_steps: int = 50

    # ---- Dynamic Ensemble Selection (DES) -----------------------------------
    use_dynamic_ensemble: bool = False
    des_algorithm: str = 'knorau'
    des_pool_classifiers: List[str] = field(
        default_factory=lambda: ['RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees']
    )
    des_dsel_fraction: float = 0.30
    des_k_neighbors: int = 7

    # ---- Conformal Prediction (H7 / H16 prerequisite) ----------------------
    use_conformal: bool = False
    conformal_alpha: float = 0.10
    conformal_calibration_fraction: float = 0.20

    # ---- Derived values (properties, not stored fields) ---------------------

    @property
    def img_size(self) -> Tuple[int, int, int]:
        """Canonical input resolution for the active CNN backbone."""
        return BACKBONE_IMG_SIZE.get(self.cnn_model, (299, 299, 3))

    @property
    def fine_tune_from_layer(self) -> int:
        """Index of the first layer to unfreeze during fine-tuning."""
        return FINE_TUNING_AT_LAYER.get(self.cnn_model, 0)

    @property
    def train_files_path(self) -> str:
        if self.use_graphic_preprocessing:
            return "./res/preprocessed_train_files.txt"
        return "./res/train_files.txt"

    @property
    def val_files_path(self) -> str:
        if self.use_graphic_preprocessing:
            return "./res/preprocessed_val_files.txt"
        return "./res/val_files.txt"

    @property
    def test_files_path(self) -> str:
        if self.use_graphic_preprocessing:
            return "./res/preprocessed_test_files.txt"
        return "./res/test_files.txt"


# ---------------------------------------------------------------------------
# Single global instance — mutated in-place by apply_cli_overrides()
# ---------------------------------------------------------------------------
cfg = Config()


# ---------------------------------------------------------------------------
# CLI override
# ---------------------------------------------------------------------------

def apply_cli_overrides(args) -> None:
    """
    Apply an argparse Namespace to the global cfg object in-place.

    Only fields whose CLI value is not None are applied, so absent flags
    leave the default untouched.  Derived properties (img_size, data paths)
    recompute automatically on next access.
    """
    _field_map = {
        # str
        'cnn_model':                 'cnn_model',
        'classifier':                'classical_classifier_model',
        'des_algorithm':             'des_algorithm',
        # int
        'batch_size':                'batch_size',
        'num_epochs':                'num_epochs',
        'num_kfolds':                'num_kfolds',
        'num_iterations':            'num_iterations',
        'num_final_models':          'num_final_models',
        'tta_n_steps':               'tta_n_steps',
        'mc_dropout_steps':          'mc_dropout_steps',
        'warmup_epochs':             'warmup_epochs',
        # float
        'label_smoothing':           'label_smoothing',
        'weight_decay':              'weight_decay',
        'cutmix_alpha':              'cutmix_alpha',
        'conformal_alpha':           'conformal_alpha',
        'conformal_calibration_fraction': 'conformal_calibration_fraction',
        # bool  (BooleanOptionalAction yields None when flag is absent)
        'use_fine_tuning':           'use_fine_tuning',
        'use_data_augmentation':     'use_data_augmentation',
        'use_feature_augmentation':  'use_feature_augmentation',
        'use_feature_preprocessing': 'use_feature_preprocessing',
        'use_graphic_preprocessing': 'use_graphic_preprocessing',
        'use_metadata':              'use_metadata',
        'use_hair_removal':          'use_hair_removal',
        'use_enhanced_contrast':     'use_enhanced_contrast',
        'use_color_normalization':   'use_color_normalization',
        'use_focal_loss':            'use_focal_loss',
        'use_tta':                   'use_tta',
        'use_mixup':                 'use_mixup',
        'use_cutmix':                'use_cutmix',
        'use_mc_dropout':            'use_mc_dropout',
        'use_dynamic_ensemble':      'use_dynamic_ensemble',
        'use_conformal':             'use_conformal',
    }
    for arg_attr, cfg_attr in _field_map.items():
        val = getattr(args, arg_attr, None)
        if val is not None:
            setattr(cfg, cfg_attr, val)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Raise ValueError when the current cfg state is inconsistent."""
    if cfg.cnn_model not in VALID_BACKBONES:
        raise ValueError(
            f"Unknown cnn_model '{cfg.cnn_model}'. Valid: {VALID_BACKBONES}"
        )
    if cfg.classical_classifier_model not in VALID_CLASSIFIERS:
        raise ValueError(
            f"Unknown classical_classifier_model '{cfg.classical_classifier_model}'. "
            f"Valid: {VALID_CLASSIFIERS}"
        )
    if cfg.des_algorithm not in VALID_DES_ALGORITHMS:
        raise ValueError(
            f"Unknown des_algorithm '{cfg.des_algorithm}'. "
            f"Valid: {VALID_DES_ALGORITHMS}"
        )
    for clf in cfg.des_pool_classifiers:
        if clf not in VALID_CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier '{clf}' in des_pool_classifiers. "
                f"Valid: {VALID_CLASSIFIERS}"
            )
    if cfg.use_color_normalization and not os.path.exists(cfg.color_norm_stats_path):
        raise ValueError(
            f"use_color_normalization=True but stats file not found: "
            f"{cfg.color_norm_stats_path}"
        )
    if cfg.use_hair_removal and cfg.use_graphic_preprocessing:
        if not os.path.exists('./results/chimera/best_weights.h5'):
            raise ValueError(
                "use_hair_removal=True but weights not found at "
                "results/chimera/best_weights.h5"
            )
