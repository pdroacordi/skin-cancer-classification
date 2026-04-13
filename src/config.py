NUM_CLASSES = 7  # Number of skin lesion classes in HAM10000
BATCH_SIZE = 16  # Smaller batch size to prevent memory issues
NUM_EPOCHS = 100

# Canonical input size per backbone.
# NOTE: changing CNN_MODEL invalidates any cached feature .npz files — delete them
# before re-running feature extraction so they are recomputed at the correct size.
BACKBONE_IMG_SIZE = {
    'VGG19':       (224, 224, 3),  # VGG19 pretrained at 224×224
    'ResNet':      (224, 224, 3),  # ResNet50 pretrained at 224×224
    'Inception':   (299, 299, 3),  # InceptionV3 pretrained at 299×299
    'Xception':    (299, 299, 3),  # Xception pretrained at 299×299
    'EfficientNet': (380, 380, 3), # EfficientNetB4 pretrained at 380×380
}

# GPU memory management
GPU_MEMORY_LIMIT = 0.9  # Fraction of GPU memory to use

# Pipeline configuration
USE_GRAPHIC_PREPROCESSING = False    # Apply hair removal, contrast enhancement, etc.
USE_DATA_AUGMENTATION     = True    # Apply data augmentation during training
USE_FEATURE_AUGMENTATION  = False
USE_FEATURE_PREPROCESSING = True    # Apply feature pre-processing
USE_FINE_TUNING           = True     # Fine-tune pretrained CNN
USE_METADATA              = False    # use Metadata (age, location, etc)

USE_HAIR_REMOVAL          = True
USE_ENHANCED_CONTRAST     = False
USE_COLOR_NORMALIZATION   = False  # Reinhard LAB color normalization (novel for HAM10000)
COLOR_NORM_STATS_PATH     = "./res/color_norm_stats.joblib"  # fitted on train images only

# Loss function configuration
USE_FOCAL_LOSS  = False  # Replace categorical_crossentropy with focal loss
LABEL_SMOOTHING = 0.0    # Set to 0.1 to regularize overconfident predictions

# Inference-time strategies
USE_TTA     = False  # Test-Time Augmentation: average N augmented predictions (+1-3 macro-F1)
TTA_N_STEPS = 8      # Number of augmented copies to average during TTA

# Training-time strategies
USE_MIXUP = False  # Mixup augmentation (Zhang et al., 2018, ICLR); blends image+label pairs

# Uncertainty quantification
USE_MC_DROPOUT   = False  # Monte Carlo Dropout inference (Gal & Ghahramani, 2016)
MC_DROPOUT_STEPS = 50     # Number of stochastic forward passes for uncertainty estimate

FINE_TUNING_AT_LAYER = {         # Layer index to start fine-tuning from
    'VGG19': 15,
    'Inception': 280,
    'ResNet': 140,
    'Xception': 100,
    'EfficientNet': 342,  # block6a_expand_conv — fine-tune last two blocks
}

VISUALIZE = False                # Display processed images for debugging

# Cross-validation
NUM_KFOLDS       = 5             # Number of folds for cross-validation
NUM_ITERATIONS   = 2             # Number of iterations for cross-validation
NUM_FINAL_MODELS = 10            # Number of final models to train

# Model selection
CNN_MODEL                  = 'ResNet'       # Options: 'VGG19', 'Inception', 'ResNet', 'Xception', 'EfficientNet'
CLASSICAL_CLASSIFIER_MODEL = 'ExtraTrees'  # Options: 'RandomForest', 'XGBoost', 'LightGBM', 'HistGradientBoosting', 'ExtraTrees', 'SVM'

# Dynamic Ensemble Selection (DESlib integration)
# When USE_DYNAMIC_ENSEMBLE=True, the feature-extraction pipeline trains a pool
# of classifiers and uses DESlib for local classifier selection at prediction
# time instead of a single CLASSICAL_CLASSIFIER_MODEL.
USE_DYNAMIC_ENSEMBLE    = False
DES_ALGORITHM           = 'knorau'    # 'knorau' | 'desmi' | 'metades' | 'singlebest'
DES_POOL_CLASSIFIERS    = ['RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees']
DES_DSEL_FRACTION       = 0.30        # Fraction of training data reserved for DES calibration
DES_K_NEIGHBORS         = 7          # DSEL neighborhood size (7 = one neighbor per class)

# IMG_SIZE derived from CNN_MODEL so each backbone uses its canonical resolution.
IMG_SIZE = BACKBONE_IMG_SIZE.get(CNN_MODEL, (299, 299, 3))

# Paths
RESULTS_DIR      = './results'
TRAIN_FILES_PATH = "./res/preprocessed_train_files.txt" if USE_GRAPHIC_PREPROCESSING else "./res/train_files.txt"
VAL_FILES_PATH   = "./res/preprocessed_val_files.txt" if USE_GRAPHIC_PREPROCESSING else"./res/val_files.txt"
TEST_FILES_PATH  = "./res/preprocessed_test_files.txt" if USE_GRAPHIC_PREPROCESSING else"./res/test_files.txt"
METADATA_PATH    = "./res/metadata.csv"

# Callbacks configuration
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5


def apply_cli_overrides(args):
    """Apply an argparse Namespace to this module's globals in-place.

    Only fields whose value is not None are applied, so absent CLI flags
    leave the default untouched.  Derived values (IMG_SIZE, data paths)
    are recomputed after all overrides are applied.
    """
    import sys
    m = sys.modules[__name__]

    _fields = {
        # str
        'cnn_model':                'CNN_MODEL',
        'classifier':               'CLASSICAL_CLASSIFIER_MODEL',
        # int
        'batch_size':               'BATCH_SIZE',
        'num_epochs':               'NUM_EPOCHS',
        'num_kfolds':               'NUM_KFOLDS',
        'num_iterations':           'NUM_ITERATIONS',
        'num_final_models':         'NUM_FINAL_MODELS',
        'tta_n_steps':              'TTA_N_STEPS',
        'mc_dropout_steps':         'MC_DROPOUT_STEPS',
        # float
        'label_smoothing':          'LABEL_SMOOTHING',
        # bool  (BooleanOptionalAction yields None when flag is absent)
        'use_fine_tuning':          'USE_FINE_TUNING',
        'use_data_augmentation':    'USE_DATA_AUGMENTATION',
        'use_feature_augmentation': 'USE_FEATURE_AUGMENTATION',
        'use_feature_preprocessing':'USE_FEATURE_PREPROCESSING',
        'use_graphic_preprocessing':'USE_GRAPHIC_PREPROCESSING',
        'use_metadata':             'USE_METADATA',
        'use_hair_removal':         'USE_HAIR_REMOVAL',
        'use_enhanced_contrast':    'USE_ENHANCED_CONTRAST',
        'use_color_normalization':  'USE_COLOR_NORMALIZATION',
        'use_focal_loss':           'USE_FOCAL_LOSS',
        'use_tta':                  'USE_TTA',
        'use_mixup':                'USE_MIXUP',
        'use_mc_dropout':           'USE_MC_DROPOUT',
        'use_dynamic_ensemble':     'USE_DYNAMIC_ENSEMBLE',
        'des_algorithm':            'DES_ALGORITHM',
    }

    for arg_attr, cfg_name in _fields.items():
        val = getattr(args, arg_attr, None)
        if val is not None:
            setattr(m, cfg_name, val)

    # Recompute derived values that depend on the overridden constants.
    m.IMG_SIZE = m.BACKBONE_IMG_SIZE.get(m.CNN_MODEL, (299, 299, 3))
    if m.USE_GRAPHIC_PREPROCESSING:
        m.TRAIN_FILES_PATH = './res/preprocessed_train_files.txt'
        m.VAL_FILES_PATH   = './res/preprocessed_val_files.txt'
        m.TEST_FILES_PATH  = './res/preprocessed_test_files.txt'
    else:
        m.TRAIN_FILES_PATH = './res/train_files.txt'
        m.VAL_FILES_PATH   = './res/val_files.txt'
        m.TEST_FILES_PATH  = './res/test_files.txt'


def validate_config():
    """Raise AssertionError when the current config state is inconsistent."""
    import os
    valid_backbones   = list(BACKBONE_IMG_SIZE)
    valid_classifiers = ['RandomForest', 'XGBoost', 'LightGBM', 'HistGradientBoosting', 'ExtraTrees', 'SVM']

    assert CNN_MODEL in valid_backbones, (
        f"Unknown CNN_MODEL '{CNN_MODEL}'. Valid options: {valid_backbones}"
    )
    assert CNN_MODEL in FINE_TUNING_AT_LAYER, (
        f"No FINE_TUNING_AT_LAYER entry for '{CNN_MODEL}'."
    )
    assert CLASSICAL_CLASSIFIER_MODEL in valid_classifiers, (
        f"Unknown CLASSICAL_CLASSIFIER_MODEL '{CLASSICAL_CLASSIFIER_MODEL}'. "
        f"Valid options: {valid_classifiers}"
    )
    valid_des_algorithms = ['knorau', 'desmi', 'metades', 'singlebest']
    assert DES_ALGORITHM in valid_des_algorithms, (
        f"Unknown DES_ALGORITHM '{DES_ALGORITHM}'. "
        f"Valid options: {valid_des_algorithms}"
    )
    valid_pool_classifiers = set(valid_classifiers)
    for clf in DES_POOL_CLASSIFIERS:
        assert clf in valid_pool_classifiers, (
            f"Unknown classifier '{clf}' in DES_POOL_CLASSIFIERS. "
            f"Valid options: {valid_classifiers}"
        )
    if USE_COLOR_NORMALIZATION:
        assert os.path.exists(COLOR_NORM_STATS_PATH), (
            f"USE_COLOR_NORMALIZATION=True but stats file not found: {COLOR_NORM_STATS_PATH}"
        )
    if USE_HAIR_REMOVAL and USE_GRAPHIC_PREPROCESSING:
        assert os.path.exists('./results/chimera/best_weights.h5'), (
            "USE_HAIR_REMOVAL=True but weights not found at results/chimera/best_weights.h5"
        )