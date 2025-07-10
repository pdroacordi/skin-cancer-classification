NUM_CLASSES = 7  # Number of skin lesion classes in HAM10000
BATCH_SIZE = 16  # Smaller batch size to prevent memory issues
NUM_EPOCHS = 100
IMG_SIZE = (299, 299, 3)  # Width, height, channels

# GPU memory management
GPU_MEMORY_LIMIT = 0.9  # Fraction of GPU memory to use

# Pipeline configuration
USE_GRAPHIC_PREPROCESSING = False    # Apply hair removal, contrast enhancement, etc.
USE_DATA_AUGMENTATION     = True    # Apply data augmentation during training
USE_FEATURE_AUGMENTATION  = False
USE_FEATURE_PREPROCESSING = False    # Apply feature pre-processing
USE_FINE_TUNING           = True     # Fine-tune pretrained CNN
USE_METADATA              = True    # use Metadata (age, location, etc)

USE_HAIR_REMOVAL          = True
USE_ENHANCED_CONTRAST     = False

FINE_TUNING_AT_LAYER = {         # Layer index to start fine-tuning from
    'VGG19': 15,
    'Inception': 280,
    'ResNet': 140,
    'Xception': 100
}

VISUALIZE = False                # Display processed images for debugging

# Cross-validation
NUM_KFOLDS       = 5             # Number of folds for cross-validation
NUM_ITERATIONS   = 2             # Number of iterations for cross-validation
NUM_FINAL_MODELS = 10            # Number of final models to train

# Model selection
CNN_MODEL                  = 'ResNet'         # Options: 'VGG19', 'Inception', 'ResNet', 'Xception'
CLASSICAL_CLASSIFIER_MODEL = 'RandomForest'  # Options: 'RandomForest', 'XGBoost', 'AdaBoost', 'ExtraTrees', 'SVM'

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