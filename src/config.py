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
CNN_MODEL                  = 'ResNet'         # Options: 'VGG19', 'Inception', 'ResNet', 'Xception', 'EfficientNet'
CLASSICAL_CLASSIFIER_MODEL = 'ExtraTrees'  # Options: 'RandomForest', 'XGBoost', 'AdaBoost', 'ExtraTrees', 'SVM'

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