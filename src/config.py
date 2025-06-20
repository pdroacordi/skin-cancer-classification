NUM_CLASSES = 7  # Number of skin lesion classes in HAM10000
BATCH_SIZE = 16  # Smaller batch size to prevent memory issues
NUM_EPOCHS = 100
IMG_SIZE = (299, 299, 3)  # Width, height, channels

# GPU memory management
GPU_MEMORY_LIMIT = 0.9  # Fraction of GPU memory to use

# Pipeline configuration
USE_GRAPHIC_PREPROCESSING = True    # Apply hair removal, contrast enhancement, etc.
USE_DATA_AUGMENTATION     = True    # Apply data augmentation during training
USE_FEATURE_AUGMENTATION  = False
USE_DATA_PREPROCESSING    = False    # Apply data pre-processing, such as PCA, SMOTE, etc.
USE_FINE_TUNING           = True     # Fine-tune pretrained CNN

USE_HAIR_REMOVAL          = True
USE_IMAGE_SEGMENTATION    = False
USE_ENHANCED_CONTRAST     = False

FINE_TUNING_AT_LAYER = {         # Layer index to start fine-tuning from
    'VGG19': 15,
    'Inception': 280,
    'ResNet': 140,
    'Xception': 100
}

VISUALIZE = False                # Display processed images for debugging

# Cross-validation
NUM_KFOLDS = 5                   # Number of folds for cross-validation
NUM_ITERATIONS = 2               # Number of iterations for cross-validation
NUM_FINAL_MODELS = 10            # Number of final models to train

# Feature extraction
NUM_PCA_COMPONENTS = None        # PCA components for dimensionality reduction (None = no PCA)

# Model selection
CNN_MODEL                  = 'VGG19'         # Options: 'VGG19', 'Inception', 'ResNet', 'Xception'
CLASSICAL_CLASSIFIER_MODEL = 'ExtraTrees'  # Options: 'RandomForest', 'XGBoost', 'AdaBoost', 'ExtraTrees', 'SVM'
CLASSIFIER_APPROACH        = "class_weight"  # Options: "class_weight", "smote", "undersampling", "hybrid"

# Paths
RESULTS_DIR      = './results'
TRAIN_FILES_PATH = "./res/train_files.txt"
VAL_FILES_PATH   = "./res/val_files.txt"
TEST_FILES_PATH  = "./res/test_files.txt"

# Callbacks configuration
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5