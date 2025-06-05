"""
CNN feature extraction + classical ML pipeline.
Handles feature extraction, model training, and evaluation.
"""

import gc
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.backend import clear_session

sys.path.append('..')
from config import (
    BATCH_SIZE,
    USE_GRAPHIC_PREPROCESSING,
    USE_DATA_PREPROCESSING,
    USE_FINE_TUNING,
    NUM_KFOLDS,
    CNN_MODEL,
    CLASSICAL_CLASSIFIER_MODEL,
    NUM_PCA_COMPONENTS,
    RESULTS_DIR,
    USE_DATA_AUGMENTATION,
    VISUALIZE,
    IMG_SIZE,
    CLASSIFIER_APPROACH,
    USE_HAIR_REMOVAL,
    USE_IMAGE_SEGMENTATION,
    USE_ENHANCED_CONTRAST,
    NUM_ITERATIONS,
    USE_FEATURE_AUGMENTATION,
    NUM_FINAL_MODELS
)

from utils.data_loaders import load_paths_labels, resize_image
from utils.graphic_preprocessing import apply_graphic_preprocessing
from models.cnn_models import get_feature_extractor_model, get_feature_extractor_from_cnn
from models.classical_models import create_ml_pipeline, tune_hyperparameters, get_default_param_grid, save_model
from utils.data_preprocessing import apply_data_preprocessing
from utils.fold_utils import save_fold_results


def setup_gpu_memory():
    """Set up GPU memory growth to avoid OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth must be set before GPUs have been initialized
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")


def create_feature_extraction_directories(base_dir=RESULTS_DIR, cnn_model_name=CNN_MODEL,
                                          classifier_name=CLASSICAL_CLASSIFIER_MODEL,
                                          num_iterations=NUM_ITERATIONS, num_folds=NUM_KFOLDS):
    """
    Create standardized directory structure for feature extraction pipeline.

    Args:
        base_dir (str): Base results directory
        cnn_model_name (str): CNN model name used as extractor
        classifier_name (str): Classical classifier name
        num_iterations (int): Number of cross-validation iterations
        num_folds (int): Number of folds per iteration

    Returns:
        dict: Dictionary with paths to created directories
    """
    # Build feature extraction path components based on configuration
    str_hair = "hair_removal_" if USE_HAIR_REMOVAL else ""
    str_contrast = "contrast_" if USE_ENHANCED_CONTRAST else ""
    str_segmented = "segmentation_" if USE_IMAGE_SEGMENTATION else ""
    str_graphic = f"{str_segmented}{str_contrast}{str_hair}" if USE_GRAPHIC_PREPROCESSING else ""
    str_augment = "use_augmentation_" if USE_DATA_AUGMENTATION else ""
    str_preprocess = f"use_data_preprocess_{CLASSIFIER_APPROACH}" if USE_DATA_PREPROCESSING else ""
    str_feature = f"use_feature_augmentation_" if USE_FEATURE_AUGMENTATION else ""

    # Create main result directory path
    result_dir = os.path.join(base_dir,
                              f"feature_extraction_{cnn_model_name}_{str_graphic}{str_augment}{str_feature}{str_preprocess}")

    # Create base directory
    os.makedirs(result_dir, exist_ok=True)

    # Initialize directory structure dictionary
    dirs = {
        'base': result_dir,
        'models': os.path.join(result_dir, "models"),
        'features': os.path.join(result_dir, "features"),
        'classifiers': {},  # Will hold classifier-specific directories
        'features_by_fold': os.path.join(result_dir, "features_by_fold")  # For cross-validation feature storage
    }

    # Create base directories
    os.makedirs(dirs['models'], exist_ok=True)
    os.makedirs(dirs['features'], exist_ok=True)
    os.makedirs(dirs['features_by_fold'], exist_ok=True)

    # Set path for feature extractor model
    dirs['extractor'] = os.path.join(dirs['models'], f"{cnn_model_name.lower()}_feature_extractor.h5")

    # Set paths for combined features
    dirs['all_features'] = os.path.join(dirs['features'], "all_features.npz")
    dirs['test_features'] = os.path.join(dirs['features'], "test_features.npz")

    # Create classifier-specific directory structure
    classifier_dir = os.path.join(result_dir, classifier_name.lower())
    os.makedirs(classifier_dir, exist_ok=True)

    # Create plots directory for the classifier
    classifier_plots_dir = os.path.join(classifier_dir, "plots")
    os.makedirs(classifier_plots_dir, exist_ok=True)

    # Initialize classifier dictionary
    classifier_dirs = {
        'base': classifier_dir,
        'plots': classifier_plots_dir,
        'iterations': {},
        'final_model': os.path.join(classifier_dir, "final_model")
    }

    # Create iteration and fold directories
    for iteration in range(1, num_iterations + 1):
        iteration_dir = os.path.join(classifier_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        os.makedirs(os.path.join(iteration_dir, "models"), exist_ok=True)

        # Store iteration paths
        iteration_dirs = {
            'base': iteration_dir,
            'models': os.path.join(iteration_dir, "models"),
            'folds': {}
        }

        # Create fold directories for this iteration
        for fold in range(1, num_folds + 1):
            fold_dir = os.path.join(iteration_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            # Create fold subdirectories
            fold_features_dir = os.path.join(fold_dir, "features")
            fold_models_dir = os.path.join(fold_dir, "models")

            os.makedirs(fold_features_dir, exist_ok=True)
            os.makedirs(fold_models_dir, exist_ok=True)

            # Store fold paths
            fold_dirs = {
                'base': fold_dir,
                'features': fold_features_dir,
                'models': fold_models_dir
            }

            # Add fold to iteration
            iteration_dirs['folds'][fold] = fold_dirs

        # Add iteration to classifier
        classifier_dirs['iterations'][iteration] = iteration_dirs

    # Add classifier to main directories
    dirs['classifiers'][classifier_name] = classifier_dirs

    # Also create features_by_fold/iteration_X directories for cross-validation feature storage
    for iteration in range(1, num_iterations + 1):
        iter_features_dir = os.path.join(dirs['features_by_fold'], f"iteration_{iteration}")
        os.makedirs(iter_features_dir, exist_ok=True)

    return dirs

def extract_features(feature_extractor, images, batch_size=32):
    """
    Extract features from images using a feature extractor model.

    Args:
        feature_extractor: Feature extractor model.
        images (numpy.array): Input images.
        batch_size (int): Batch size for feature extraction.

    Returns:
        numpy.array: Extracted features.
    """
    # Extract features in batches to avoid memory issues
    num_samples = len(images)
    num_batches = int(np.ceil(num_samples / batch_size))

    features_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_images = images[start_idx:end_idx]
        batch_features = feature_extractor.predict(batch_images, verbose=0)

        features_list.append(batch_features)

    # Concatenate all features
    features = np.concatenate(features_list, axis=0)
    features = features.astype(np.float32)

    return features


def extract_and_save_features(feature_extractor, paths, labels=None,
                              preprocess_fn=None, model_name=CNN_MODEL,
                              features_save_path=None, apply_augmentation=USE_FEATURE_AUGMENTATION):
    """
    Extract features and save them to disk with augmentation metadata.

    Args:
        feature_extractor: Feature extractor model.
        paths (numpy.array): Image paths.
        labels (numpy.array, optional): Image labels.
        preprocess_fn (callable, optional): Function for image preprocessing.
        model_name (str): CNN model name for preprocessing.
        features_save_path (str, optional): Path to save features.
        apply_augmentation (bool): Whether to apply data augmentation.

    Returns:
        tuple: (features, labels) if labels is provided, otherwise just features.
    """
    print(f"Extracting features from {len(paths)} images...")
    print(f"Feature augmentation: {'ENABLED' if apply_augmentation else 'DISABLED'}")

    try:
        # Extract features
        if labels is not None:
            features, labels_out = extract_features_from_paths(
                feature_extractor=feature_extractor,
                paths=paths,
                labels=labels,
                preprocess_fn=preprocess_fn,
                model_name=model_name,
                apply_augmentation=apply_augmentation
            )

            # Calculate augmentation factor if applicable
            if apply_augmentation and len(features) > len(labels):
                if len(features) % len(labels) == 0:
                    augmentation_factor = len(features) // len(labels)
                    print(f"Features extracted with {augmentation_factor}x augmentation.")
                else:
                    print(
                        f"Warning: Feature count ({len(features)}) is not a clean multiple of label count ({len(labels)})")
                    augmentation_factor = None
            else:
                augmentation_factor = 1  # No effective augmentation
        else:
            features = extract_features_from_paths(
                feature_extractor=feature_extractor,
                paths=paths,
                labels=None,
                preprocess_fn=preprocess_fn,
                model_name=model_name,
                apply_augmentation=apply_augmentation
            )
            labels_out = None
            augmentation_factor = None

        # Ensure features are in the correct format
        if features is None or len(features) == 0:
            print("WARNING: No features were extracted! This will cause errors.")
            # Return empty arrays as a fallback instead of None
            empty_shape = feature_extractor.output_shape[1:]
            features = np.array([]).reshape((0, *empty_shape))
            if labels is not None:
                return features, labels
            return features

        features = features.astype(np.float32)

        # Save features if path was provided
        if features_save_path:
            print(f"Saving features to: {features_save_path}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(features_save_path), exist_ok=True)

            # Determine format based on extension
            if features_save_path.endswith('.npz'):
                # Save with augmentation metadata
                if labels_out is not None:
                    np.savez(
                        features_save_path,
                        features=features,
                        labels=labels_out,
                        augmentation_enabled=apply_augmentation,
                        augmentation_factor=augmentation_factor if augmentation_factor is not None else 1,
                        feature_augmentation_flag='USE_FEATURE_AUGMENTATION'  # Track which flag was used
                    )
                else:
                    np.savez(
                        features_save_path,
                        features=features,
                        augmentation_enabled=apply_augmentation,
                        augmentation_factor=augmentation_factor if augmentation_factor is not None else 1,
                        feature_augmentation_flag='USE_FEATURE_AUGMENTATION'
                    )
            else:
                # Default to .npy if extension is not specified
                if not features_save_path.endswith('.npy'):
                    features_save_path += '.npy'
                np.save(features_save_path, features)

        if labels is not None:
            return features, labels_out if labels_out is not None else labels
        else:
            return features

    except Exception as e:
        print(f"ERROR in extract_and_save_features: {e}")
        import traceback
        traceback.print_exc()
        # Handle the error gracefully by returning empty arrays
        empty_shape = feature_extractor.output_shape[1:]
        features = np.array([]).reshape((0, *empty_shape))
        if labels is not None:
            return features, labels
        return features

def extract_features_from_paths(feature_extractor, paths, labels=None,
                                preprocess_fn=None, model_name=CNN_MODEL,
                                batch_size=BATCH_SIZE, apply_augmentation=USE_FEATURE_AUGMENTATION):
    """
    Extract features from image paths using a feature extractor model.
    Memory-efficient version to avoid GPU OOM errors.

    Args:
        feature_extractor: Feature extractor model.
        paths (numpy.array): Image paths.
        labels (numpy.array, optional): Image labels.
        preprocess_fn (callable, optional): Function for image preprocessing.
        model_name (str): CNN model name for preprocessing.
        batch_size (int): Batch size for feature extraction.
        apply_augmentation (bool): Whether to apply image augmentation.
    Returns:
        tuple: (features, labels) if labels is provided, otherwise just features.
    """
    print(f"Loading and preprocessing {len(paths)} images with batch_size={batch_size}...")
    print(f"Feature augmentation: {'ENABLED' if apply_augmentation else 'DISABLED'}")

    # Create a preprocessing function for each image
    def process_image(path, augment=False, aug_index=None):
        import cv2
        from utils.data_loaders import apply_model_preprocessing

        # Load image
        image = cv2.imread(path)
        if image is None:
            print(f"Error loading image: {path}")
            return None

        image = resize_image(image, IMG_SIZE[:2])

        # Apply preprocessing if available
        if preprocess_fn:
            image = preprocess_fn(image)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation if requested
        if augment and augmentation_pipelines and aug_index is not None:
            # Use specific augmentation pipeline from the list
            pipeline = augmentation_pipelines[aug_index]
            image = pipeline(image=image)['image']

        # Apply model-specific preprocessing
        image = apply_model_preprocessing(image, model_name)

        return image

    # Process images in batches with additional error handling
    num_samples = len(paths)
    num_batches = int(np.ceil(num_samples / batch_size))

    all_features = []
    all_labels = []

    augmentation_pipelines = None
    if apply_augmentation:
        from utils.augmentation import AugmentationFactory
        augmentation_pipelines = AugmentationFactory.get_feature_extraction_augmentation()
        num_augmentations = len(augmentation_pipelines)
        print(f"Using {num_augmentations} augmentation pipelines for feature extraction.")
    else:
        num_augmentations = 1
        print("No augmentation applied during feature extraction.")

    # Set up a smaller mini-batch size for prediction to avoid OOM errors
    mini_batch_size = min(4, batch_size)  # Use mini-batches of at most 4 images

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        print(f"Processing batch {i + 1}/{num_batches} (images {start_idx + 1}-{end_idx} of {num_samples})")

        batch_paths = paths[start_idx:end_idx]

        # Process batch of images
        batch_images = []
        batch_indices = []

        for j, path in enumerate(batch_paths):
            try:
                image = process_image(path)
                if image is not None:
                    batch_images.append(image)
                    batch_indices.append(start_idx + j)

                    # Add augmented versions if requested
                    if apply_augmentation and augmentation_pipelines:
                        # Skip index 0 as it's the identity transformation (no augmentation)
                        for aug_index in range(1, num_augmentations):
                            aug_image = process_image(path, augment=True, aug_index=aug_index)
                            if aug_image is not None:
                                batch_images.append(aug_image)
                                batch_indices.append(start_idx + j)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue

        if not batch_images:
            print(f"No valid images in batch {i + 1}, skipping.")
            continue

        # Process images in mini-batches to avoid OOM
        mini_batch_features = []
        num_mini_batches = int(np.ceil(len(batch_images) / mini_batch_size))

        for k in range(num_mini_batches):
            mini_start = k * mini_batch_size
            mini_end = min((k + 1) * mini_batch_size, len(batch_images))
            mini_images = batch_images[mini_start:mini_end]

            try:
                # Convert to numpy array
                mini_images_array = np.array(mini_images)

                # Extract features for this mini-batch
                mini_features = feature_extractor.predict(mini_images_array, verbose=0)
                mini_batch_features.append(mini_features)

                # Clean up to free memory
                del mini_images_array, mini_features
                gc.collect()
                tf.keras.backend.clear_session()

            except tf.errors.ResourceExhaustedError:
                print(f"OOM error in mini-batch {k + 1}. Trying with single images...")

                # If mini-batch fails, try with individual images
                for idx, img in enumerate(mini_images):
                    try:
                        img_array = np.expand_dims(img, axis=0)
                        single_feature = feature_extractor.predict(img_array, verbose=0)
                        mini_batch_features.append(single_feature)

                        # Clean up
                        del img_array, single_feature
                        gc.collect()
                        tf.keras.backend.clear_session()

                    except Exception as e2:
                        print(f"    Error processing single image {mini_start + idx}: {e2}")
                        continue

            except Exception as e:
                print(f"  Error in mini-batch {k + 1}: {e}")
                continue

        # Combine mini-batch features if any were successfully extracted
        if mini_batch_features:
            try:
                combined_features = np.vstack(mini_batch_features)
                all_features.append(combined_features)

                # Add labels if provided
                if labels is not None:
                    # Get unique indices and count number of features per index (including augmentations)
                    unique_indices = np.unique(batch_indices)
                    if apply_augmentation:
                        # Count how many features were actually extracted for each index
                        feature_count_per_index = 0
                        if len(combined_features) > 0:
                            feature_count_per_index = len(combined_features) / len(unique_indices)

                        # Repeat each label based on actual extracted features
                        batch_labels = np.repeat(labels[unique_indices], feature_count_per_index)
                    else:
                        batch_labels = labels[batch_indices[:len(combined_features)]]

                    all_labels.append(batch_labels)

                # Clean up
                del mini_batch_features, combined_features
                gc.collect()

            except Exception as e:
                print(f"Error combining features from batch {i + 1}: {e}")
                # Just skip this batch if combining fails
                continue

        # Force garbage collection after each batch
        gc.collect()
        tf.keras.backend.clear_session()

    # Concatenate results if any were successfully extracted
    if all_features:
        try:
            features = np.concatenate(all_features, axis=0)
            features = features.astype(np.float32)

            if labels is not None and all_labels:
                # Make sure labels and features have the same length
                labels_out = np.concatenate(all_labels, axis=0)
                if len(labels_out) > len(features):
                    labels_out = labels_out[:len(features)]
                elif len(labels_out) < len(features):
                    # This shouldn't happen with the fixes above, but just in case
                    print(f"Warning: Labels count ({len(labels_out)}) doesn't match features count ({len(features)})")
                    features = features[:len(labels_out)]

                return features, labels_out

            return features
        except Exception as e:
            print(f"Error concatenating features: {e}")
            empty_shape = feature_extractor.output_shape[1:]
            return np.array([]).reshape((0, *empty_shape)), np.array([]) if labels is not None else np.array([])
    else:
        print("No features were successfully extracted.")
        empty_shape = feature_extractor.output_shape[1:]
        return np.array([]).reshape((0, *empty_shape)), np.array([]) if labels is not None else np.array([])


def load_or_extract_features(feature_extractor, paths, labels=None,
                             preprocess_fn=None, model_name=CNN_MODEL,
                             features_save_path=None, apply_augmentation=USE_FEATURE_AUGMENTATION):
    """
    Load cached features or extract new ones with improved handling of augmented data.

    Args:
        feature_extractor: Feature extractor model.
        paths (numpy.array): Image paths.
        labels (numpy.array, optional): Image labels.
        preprocess_fn (callable, optional): Function for image preprocessing.
        model_name (str): CNN model name for preprocessing.
        features_save_path (str, optional): Path to save/load features.
        apply_augmentation (bool): Whether augmentation was applied or should be applied.

    Returns:
        tuple: (features, labels) if labels is provided, otherwise just features.
    """
    if features_save_path is None:
        print(f"Extracting features from {len(paths)} images...")
        return extract_features_from_paths(
            feature_extractor=feature_extractor,
            paths=paths,
            labels=labels,
            preprocess_fn=preprocess_fn,
            model_name=model_name,
            apply_augmentation=apply_augmentation
        )

        # Try to load cached features if file exists
    if os.path.exists(features_save_path):
        print(f"Loading cached features from: {features_save_path}")
        try:
            # Try to load based on file extension
            if features_save_path.endswith('.npz'):
                features_data = np.load(features_save_path, allow_pickle=True)

                # Check if it's an NPZ file with the expected keys
                if isinstance(features_data, np.lib.npyio.NpzFile) and 'features' in features_data:
                    features = features_data['features'].astype(np.float32)

                    # Check if the file contains labels
                    if 'labels' in features_data:
                        saved_labels = features_data['labels']
                    else:
                        saved_labels = None

                    # Check if the file contains augmentation metadata
                    if 'augmentation_enabled' in features_data:
                        cached_augmentation = bool(features_data['augmentation_enabled'])
                        print(f"Cache metadata: augmentation_enabled={cached_augmentation}")
                    else:
                        cached_augmentation = None  # Unknown

                    if 'augmentation_factor' in features_data:
                        cached_aug_factor = int(features_data['augmentation_factor'])
                        print(f"Cache metadata: augmentation_factor={cached_aug_factor}x")
                    else:
                        cached_aug_factor = None  # Unknown

                    # Check feature consistency
                    if features.ndim != 2:
                        print(f"Warning: Cached features have unexpected shape: {features.shape}. Re-extracting...")
                        return extract_and_save_features(
                            feature_extractor, paths, labels, preprocess_fn, model_name,
                            features_save_path, apply_augmentation
                        )

                    # Check if augmentation settings match
                    if cached_augmentation is not None and cached_augmentation != apply_augmentation:
                        print(f"Augmentation mismatch: cached={cached_augmentation}, requested={apply_augmentation}")
                        print("Re-extracting features with correct augmentation settings...")
                        return extract_and_save_features(
                            feature_extractor, paths, labels, preprocess_fn, model_name,
                            features_save_path, apply_augmentation
                        )

                    # Check if labels match (if provided)
                    if labels is not None and saved_labels is not None:
                        if len(labels) != len(saved_labels):
                            # Check if the difference could be due to augmentation
                            if len(saved_labels) > len(labels) and len(saved_labels) % len(labels) == 0:
                                detected_aug_factor = len(saved_labels) // len(labels)
                                print(f"Detected augmentation factor: {detected_aug_factor}x")

                                # Determine if this matches our current augmentation settings
                                if apply_augmentation:
                                    print(f"Current settings use augmentation, consistent with cached features.")
                                    return features, saved_labels
                                else:
                                    print("Current settings don't use augmentation, but cached features do.")
                                    print("Re-extracting features to match current settings...")
                                    return extract_and_save_features(
                                        feature_extractor, paths, labels, preprocess_fn,
                                        model_name, features_save_path, apply_augmentation
                                    )
                            else:
                                print(f"Label count mismatch: Current={len(labels)}, Cached={len(saved_labels)}")
                                print("Re-extracting features...")
                                return extract_and_save_features(
                                    feature_extractor, paths, labels, preprocess_fn,
                                    model_name, features_save_path, apply_augmentation
                                )
                        elif not np.array_equal(labels, saved_labels):
                            # Labels differ - check if it's just ordering or actual difference
                            print("Labels differ. Re-extracting features...")
                            return extract_and_save_features(
                                feature_extractor, paths, labels, preprocess_fn,
                                model_name, features_save_path, apply_augmentation
                            )
                        else:
                            print("Labels match exactly! Using cached features.")

                        # If we reach here, labels match exactly, so return features and labels
                        return features, labels
                    else:
                        # No current labels or no saved labels
                        if labels is None and saved_labels is not None:
                            print("No labels provided, using cached features and labels.")
                            return features, saved_labels
                        elif labels is not None and saved_labels is None:
                            print("Cached features have no labels. Using current labels.")
                            return features, labels
                        else:
                            print("Both current and cached have no labels.")
                            return features
                else:
                    print("Warning: NPZ file format is not as expected. Re-extracting features...")
                    return extract_and_save_features(
                        feature_extractor, paths, labels, preprocess_fn,
                        model_name, features_save_path, apply_augmentation
                    )

            elif features_save_path.endswith('.npy'):
                features = np.load(features_save_path, allow_pickle=False)
                features = features.astype(np.float32)

                # Check basic consistency
                if features.ndim != 2:
                    print(f"Warning: Cached features have unexpected shape: {features.shape}. Re-extracting...")
                    return extract_and_save_features(
                        feature_extractor, paths, labels, preprocess_fn,
                        model_name, features_save_path, apply_augmentation
                    )

                # Cannot check augmentation settings with .npy files
                print("Warning: .npy files don't store augmentation metadata. Consider using .npz format.")

                if labels is not None:
                    if len(features) != len(labels) and apply_augmentation and len(features) > len(labels):
                        if len(features) % len(labels) == 0:
                            augmentation_factor = len(features) // len(labels)
                            print(f"Detected augmentation factor in .npy file: {augmentation_factor}x")
                            print("Using cached features with current labels.")
                            return features, labels
                        else:
                            print(f"Warning: Feature/label count mismatch. Re-extracting...")
                            return extract_and_save_features(
                                feature_extractor, paths, labels, preprocess_fn,
                                model_name, features_save_path, apply_augmentation
                            )
                    elif len(features) != len(labels):
                        print(f"Warning: Feature/label count mismatch. Re-extracting...")
                        return extract_and_save_features(
                            feature_extractor, paths, labels, preprocess_fn,
                            model_name, features_save_path, apply_augmentation
                        )
                    return features, labels
                return features

            else:
                print(f"Unsupported feature file extension: {features_save_path}. Re-extracting features...")
                return extract_and_save_features(
                    feature_extractor, paths, labels, preprocess_fn,
                    model_name, features_save_path, apply_augmentation
                )

        except Exception as e:
            print(f"Error loading cached features: {e}. Re-extracting...")
            return extract_and_save_features(
                feature_extractor, paths, labels, preprocess_fn,
                model_name, features_save_path, apply_augmentation
            )

        # If we reach here, the file doesn't exist
    print(f"No cached features found at {features_save_path}. Extracting...")
    return extract_and_save_features(
        feature_extractor, paths, labels, preprocess_fn,
        model_name, features_save_path, apply_augmentation
    )


def augment_features_with_balanced_sampling(features, labels, target_count=None):
    """
    Create a balanced dataset by augmenting features from underrepresented classes.

    Args:
        features (numpy.array): Feature matrix.
        labels (numpy.array): Target labels.
        target_count (int, optional): Target count for each class. If None, uses the largest class count.

    Returns:
        tuple: (balanced_features, balanced_labels)
    """
    # Count samples in each class
    unique_classes = np.unique(labels)
    class_counts = {cls: np.sum(labels == cls) for cls in unique_classes}

    # Set target count to the largest class if not specified
    if target_count is None:
        target_count = max(class_counts.values())

    balanced_features = []
    balanced_labels = []

    # Process each class
    for cls in unique_classes:
        # Get indices for this class
        cls_indices = np.where(labels == cls)[0]
        cls_features = features[cls_indices]
        cls_labels = labels[cls_indices]

        # If class already has enough samples, just take the original samples
        if len(cls_indices) >= target_count:
            balanced_features.append(cls_features[:target_count])
            balanced_labels.append(cls_labels[:target_count])
        else:
            # Need to augment this class
            # First include all original samples
            aug_features = list(cls_features)
            aug_labels = list(cls_labels)

            # Use random oversampling with small Gaussian noise
            while len(aug_features) < target_count:
                idx = np.random.randint(0, len(cls_features))
                feat = cls_features[idx]

                # Add small Gaussian noise to create synthetic sample
                noise = np.random.normal(0, 0.01, size=feat.shape)
                synth_feat = feat + noise

                aug_features.append(synth_feat)
                aug_labels.append(cls)

            balanced_features.append(np.array(aug_features[:target_count]))
            balanced_labels.append(np.array(aug_labels[:target_count]))

    # Concatenate all classes
    balanced_features = np.concatenate(balanced_features, axis=0)
    balanced_labels = np.concatenate(balanced_labels, axis=0)

    balanced_features = balanced_features.astype(np.float32)

    # Shuffle the balanced dataset
    shuffle_indices = np.random.permutation(len(balanced_features))
    balanced_features = balanced_features[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]

    return balanced_features, balanced_labels


def train_and_evaluate_classical_model(train_features, train_labels,
                                       val_features, val_labels,
                                       classifier_name=CLASSICAL_CLASSIFIER_MODEL,
                                       use_pca=True, n_components=NUM_PCA_COMPONENTS,
                                       tune_hyperparams=True, result_dir=None,
                                       model_save_path=None, class_weights=None):
    """
    Train and evaluate a classical ML model on extracted features.

    Args:
        train_features (numpy.array): Training feature matrix.
        train_labels (numpy.array): Training labels.
        val_features (numpy.array): Validation feature matrix.
        val_labels (numpy.array): Validation labels.
        classifier_name (str): Name of the classifier.
        use_pca (bool): Whether to use PCA for dimensionality reduction.
        n_components (int, optional): Number of PCA components.
        tune_hyperparams (bool): Whether to tune hyperparameters.
        result_dir (str, optional): Directory to save results.
        model_save_path (str, optional): Path to save the trained model.
        class_weights (dict, optional): Dictionary mapping class indices to weights.

    Returns:
        tuple: (model, evaluation_results)
    """
    print(f"Training {classifier_name} classifier...")

    train_features = train_features.astype(np.float32)
    val_features = val_features.astype(np.float32)

    # Create ML pipeline
    pipeline = create_ml_pipeline(
        classifier_name=classifier_name,
        use_pca=use_pca,
        n_components=n_components
    )

    # Tune hyperparameters if requested
    if tune_hyperparams:
        print("Tuning hyperparameters...")
        param_grid = get_default_param_grid(classifier_name)

        # Add class weight options for tree-based models if class weights are provided
        if class_weights is not None and classifier_name in ["RandomForest", "ExtraTrees", "AdaBoost"]:
            if "classifier__class_weight" not in param_grid:
                param_grid["classifier__class_weight"] = ['balanced', None]

        grid_search = tune_hyperparameters(
            pipeline=pipeline,
            X=train_features,
            y=train_labels,
            param_grid=param_grid,
            cv=5,
            class_weights=class_weights
        )

        # Use best model
        model = grid_search.best_estimator_
    else:
        # If class weights are provided and we're not tuning hyperparameters
        if class_weights is not None:
            # Get the classifier from the pipeline
            if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                classifier = pipeline.named_steps['classifier']

                # Train the preprocessing steps
                if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
                    for name, transform in pipeline.steps[:-1]:  # All steps except classifier
                        if hasattr(transform, "fit"):
                            train_features = transform.fit_transform(train_features, train_labels)

                # Check if classifier supports sample weights
                if hasattr(classifier, 'fit') and 'sample_weight' in classifier.fit.__code__.co_varnames:
                    # Create sample weights from class weights
                    sample_weights = np.array([class_weights[label] for label in train_labels])
                    classifier.fit(train_features, train_labels, sample_weight=sample_weights)
                    print("Training with sample weights based on class imbalance")
                else:
                    # If sample_weight not supported, try to set class_weight attribute
                    if hasattr(classifier, 'class_weight'):
                        classifier.class_weight = class_weights
                        print("Setting class_weight parameter")
                    classifier.fit(train_features, train_labels)

                model = pipeline
            else:
                # Fall back to regular pipeline fit
                model = pipeline
                model.fit(train_features, train_labels)
        else:
            # Train model with default parameters
            model = pipeline
            model.fit(train_features, train_labels)

    # Save model if path is provided
    if model_save_path:
        save_model(model, model_save_path)

    # Evaluate on validation set
    val_pred = model.predict(val_features)

    # Calculate metrics
    report = classification_report(val_labels, val_pred, output_dict=True)

    # Print classification report
    print("\nValidation Set Classification Report:")
    print(classification_report(val_labels, val_pred))

    # Plot confusion matrix if result directory is provided
    if result_dir:
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                fi_plot_path = os.path.join(result_dir, "plots", "feature_importance.png")

                # Get feature names or indices
                if hasattr(model, 'named_steps') and 'pca' in model.named_steps:
                    # For PCA, feature names are PCA components
                    feature_names = [f"PC{i + 1}" for i in range(len(classifier.feature_importances_))]
                else:
                    # Otherwise, use feature indices
                    feature_names = [f"Feature {i + 1}" for i in range(len(classifier.feature_importances_))]

    # Evaluation results
    evaluation_results = {
        "accuracy": report["accuracy"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "class_report": report
    }

    return model, evaluation_results


def run_kfold_feature_extraction(all_paths, all_labels, dirs, class_names=None):
    """
    Run feature extraction for each fold of k-fold cross-validation.
    Extracts features separately for each fold to avoid data leakage.

    Args:
        all_paths (numpy.array): All image paths.
        all_labels (numpy.array): All labels.
        dirs (dict): Directory to save results.
        class_names (list, optional): List of class names.

    Returns:
        dict: Dictionary containing fold-specific features and paths
    """
    from sklearn.model_selection import StratifiedKFold
    from config import NUM_KFOLDS, NUM_ITERATIONS

    # Create feature extraction directory
    features_by_fold_dir = dirs['features_by_fold']
    os.makedirs(features_by_fold_dir, exist_ok=True)

    # Define preprocessing function
    preprocess_fn = None
    if USE_GRAPHIC_PREPROCESSING:
        preprocess_fn = lambda img: apply_graphic_preprocessing(
            img,
            use_hair_removal=USE_HAIR_REMOVAL,
            use_contrast_enhancement=USE_ENHANCED_CONTRAST,
            use_segmentation=USE_IMAGE_SEGMENTATION,
            visualize=False
        )

    feature_extractor, _ = get_feature_extractor_model(
        model_name=CNN_MODEL,
        fine_tune=USE_FINE_TUNING,
        save_path=dirs['extractor']  # Use the path from the directory structure
    )

    # Initialize results dictionary
    fold_features = {
        'iterations': []
    }

    # Run multiple iterations
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'=' * 50}")
        print(f"Feature Extraction: Iteration {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'=' * 50}")

        # Create iteration directory
        iter_dir = os.path.join(features_by_fold_dir, f"iteration_{iteration + 1}")
        os.makedirs(iter_dir, exist_ok=True)

        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=NUM_KFOLDS, shuffle=True, random_state=42 + iteration)

        # For storing fold information
        iteration_folds = []

        # Get integer labels if they're one-hot encoded
        if len(all_labels.shape) > 1 and all_labels.shape[1] > 1:
            stratify_labels = np.argmax(all_labels, axis=1)
        else:
            stratify_labels = all_labels

        # Run each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, stratify_labels), 1):
            print(f"\n{'=' * 40}")
            print(f"Extracting features for Iteration {iteration}, Fold {fold}/{NUM_KFOLDS}")
            print(f"{'=' * 40}")

            # Split data
            train_paths, val_paths = all_paths[train_idx], all_paths[val_idx]
            train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

            # Paths for saving features
            train_features_path = os.path.join(iter_dir, f"train_features_fold_{fold}.npz")
            val_features_path = os.path.join(iter_dir, f"val_features_fold_{fold}.npz")

            # Extract features for this fold
            train_features, train_labels_out = load_or_extract_features(
                feature_extractor=feature_extractor,
                paths=train_paths,
                labels=train_labels,
                preprocess_fn=preprocess_fn,
                model_name=CNN_MODEL,
                features_save_path=train_features_path,
                apply_augmentation=USE_FEATURE_AUGMENTATION
            )

            val_features, val_labels_out = load_or_extract_features(
                feature_extractor=feature_extractor,
                paths=val_paths,
                labels=val_labels,
                preprocess_fn=preprocess_fn,
                model_name=CNN_MODEL,
                features_save_path=val_features_path,
            )

            # Store fold information
            fold_info = {
                'fold': fold,
                'train_features_path': train_features_path,
                'val_features_path': val_features_path,
                'train_indices': train_idx,
                'val_indices': val_idx
            }
            iteration_folds.append(fold_info)

            print(f"Fold {fold} feature extraction complete.")
            print(f"Training features shape: {train_features.shape}")
            print(f"Validation features shape: {val_features.shape}")

            # Release memory
            del train_features, val_features, train_labels_out, val_labels_out
            gc.collect()

            # Store iteration information
        fold_features['iterations'].append({
            'iteration': iteration,
            'folds': iteration_folds
        })

        print("\nFeature extraction for all folds completed successfully.")

    print("\nFeature extraction for all folds completed successfully.")

    # Save fold information
    fold_info_path = os.path.join(features_by_fold_dir, "fold_features_info.json")
    import json
    with open(fold_info_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_serializable = fold_features.copy()
        for iteration in json_serializable['iterations']:
            for fold in iteration['folds']:
                fold['train_indices'] = fold['train_indices'].tolist()
                fold['val_indices'] = fold['val_indices'].tolist()

        json.dump(json_serializable, f, indent=2)

    return fold_features


def run_model_training_by_fold(fold_features, result_dir, tune_hyperparams=True, class_names=None):
    """
    Train classical models using pre-extracted features for each fold.

    Args:
        fold_features (dict): Dictionary with fold-specific feature paths from run_kfold_feature_extraction
        result_dir (str): Directory to save results
        tune_hyperparams (bool): Whether to tune hyperparameters
        class_names (list, optional): List of class names

    Returns:
        dict: Dictionary with training results
    """
    # Results container
    all_results = {
        'fold_results': [],
        'all_y_true': [],
        'all_y_pred': []
    }

    # Best model tracking
    best_model_metrics = {
        'iteration': 0,
        'fold': 0,
        'accuracy': 0,
        'macro_avg_f1': 0,
        'model_path': None,
        'hyperparameters': None
    }

    base_dir = result_dir

    # Process each iteration
    for iteration_data in fold_features['iterations']:
        iteration = iteration_data['iteration']
        folds = iteration_data['folds']

        print(f"\n{'=' * 50}")
        print(f"Training Models: Iteration {iteration+1}/{len(fold_features['iterations'])}")
        print(f"{'=' * 50}")

        iter_dir = os.path.join(base_dir, f"iteration_{iteration+1}")
        iter_plots_dir = os.path.join(iter_dir, "plots")
        os.makedirs(iter_dir, exist_ok=True)
        os.makedirs(iter_plots_dir, exist_ok=True)

        # List to store evaluation results for this iteration
        fold_results = []

        # Dictionary to collect predictions for this iteration
        iteration_y_true = []
        iteration_y_pred = []

        # Process each fold
        for fold_data in folds:
            fold = fold_data['fold']
            train_features_path = fold_data['train_features_path']
            val_features_path = fold_data['val_features_path']

            print(f"\n{'=' * 40}")
            print(f"Training: Iteration {iteration+1}, Fold {fold}/{len(folds)}")
            print(f"{'=' * 40}")

            fold_dir = os.path.join(iter_dir, f"fold_{fold}")
            fold_models_dir = os.path.join(fold_dir, "models")
            fold_plots_dir = os.path.join(fold_dir, "plots")
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(fold_models_dir, exist_ok=True)
            os.makedirs(fold_plots_dir, exist_ok=True)

            # Model save path
            model_save_path = os.path.join(fold_models_dir, f"{CLASSICAL_CLASSIFIER_MODEL.lower()}_model.joblib")

            # Load features
            print(f"Loading training features from: {train_features_path}")
            train_data = np.load(train_features_path, allow_pickle=True)
            train_features = train_data['features'].astype(np.float32)
            train_labels = train_data['labels']

            print(f"Loading validation features from: {val_features_path}")
            val_data = np.load(val_features_path, allow_pickle=True)
            val_features = val_data['features'].astype(np.float32)
            val_labels = val_data['labels']

            # Print data shapes
            print(f"Training features shape: {train_features.shape}")
            print(f"Validation features shape: {val_features.shape}")
            print(f"Training class distribution: {np.bincount(train_labels)}")
            print(f"Validation class distribution: {np.bincount(val_labels)}")

            # Apply data preprocessing if enabled
            class_weights = None
            if USE_DATA_PREPROCESSING:
                print(f"Applying data preprocessing to iteration {iteration}, fold {fold} training data...")

                if CLASSIFIER_APPROACH == "class_weight":
                    train_features, train_labels, class_weights = apply_data_preprocessing(
                        features=train_features,
                        labels=train_labels,
                        method="class_weight",
                        random_state=42 + iteration * NUM_KFOLDS + fold
                    )
                elif CLASSIFIER_APPROACH in ["hybrid", "undersampling", "smote"]:
                    train_features, train_labels, _ = apply_data_preprocessing(
                        features=train_features,
                        labels=train_labels,
                        method=CLASSIFIER_APPROACH,
                        random_state=42 + iteration * NUM_KFOLDS + fold
                    )

            # Train model
            try:
                model, evaluation = train_and_evaluate_classical_model(
                    train_features=train_features,
                    train_labels=train_labels,
                    val_features=val_features,
                    val_labels=val_labels,
                    classifier_name=CLASSICAL_CLASSIFIER_MODEL,
                    use_pca=(NUM_PCA_COMPONENTS is not None),
                    n_components=NUM_PCA_COMPONENTS,
                    tune_hyperparams=tune_hyperparams,
                    result_dir=fold_dir,
                    model_save_path=model_save_path,
                    class_weights=class_weights
                )

                # Get predictions
                val_pred = model.predict(val_features)

                # Add to iteration predictions
                iteration_y_true.extend(val_labels)
                iteration_y_pred.extend(val_pred)

                # Update best model if current is better
                if evaluation['accuracy'] > best_model_metrics['accuracy'] or \
                        (evaluation['accuracy'] == best_model_metrics['accuracy'] and
                         evaluation['macro_avg_f1'] > best_model_metrics['macro_avg_f1']):

                    # Capture hyperparameters
                    if hasattr(model, 'get_params'):
                        model_params = model.get_params()
                    else:
                        model_params = {}

                    # Store hyperparameters and configuration
                    hyperparameters = {
                        'classifier_name': CLASSICAL_CLASSIFIER_MODEL,
                        'use_pca': (NUM_PCA_COMPONENTS is not None),
                        'n_components': NUM_PCA_COMPONENTS,
                        'class_weights': class_weights,
                        'preprocessing_approach': CLASSIFIER_APPROACH if USE_DATA_PREPROCESSING else None,
                        'model_params': model_params
                    }

                    best_model_metrics = {
                        'iteration': iteration,
                        'fold': fold,
                        'accuracy': evaluation['accuracy'],
                        'macro_avg_f1': evaluation['macro_avg_f1'],
                        'model_path': model_save_path,
                        'hyperparameters': hyperparameters
                    }

                # Store fold results
                fold_result = {
                    'iteration': iteration,
                    'fold': fold,
                    'accuracy': evaluation['accuracy'],
                    'macro_avg_precision': evaluation['macro_avg_precision'],
                    'macro_avg_recall': evaluation['macro_avg_recall'],
                    'macro_avg_f1': evaluation['macro_avg_f1'],
                    'class_report': evaluation['class_report']
                }

                fold_results.append(fold_result)

            except Exception as e:
                print(f"Error in iteration {iteration}, fold {fold}: {e}")
                continue

            # Clean up
            del train_features, val_features, train_labels, val_labels, model
            gc.collect()

        # Calculate overall metrics for this iteration
        iteration_y_true = np.array(iteration_y_true)
        iteration_y_pred = np.array(iteration_y_pred)

        # Print overall classification report for this iteration
        print(f"\nOverall Iteration {iteration} Results:")
        print(classification_report(iteration_y_true, iteration_y_pred))

        # Store iteration results
        all_results['fold_results'].extend(fold_results)
        all_results['all_y_true'].extend(iteration_y_true)
        all_results['all_y_pred'].extend(iteration_y_pred)

        # Calculate average fold metrics for this iteration
        if fold_results:
            avg_accuracy = np.mean([res['accuracy'] for res in fold_results])
            avg_precision = np.mean([res['macro_avg_precision'] for res in fold_results])
            avg_recall = np.mean([res['macro_avg_recall'] for res in fold_results])
            avg_f1 = np.mean([res['macro_avg_f1'] for res in fold_results])

            print(f"\nIteration {iteration} Average Metrics:")
            print(f"Accuracy: {avg_accuracy:.4f}")
            print(f"Precision: {avg_precision:.4f}")
            print(f"Recall: {avg_recall:.4f}")
            print(f"F1 Score: {avg_f1:.4f}")

    # Calculate overall metrics across all iterations
    all_y_true = np.array(all_results['all_y_true'])
    all_y_pred = np.array(all_results['all_y_pred'])

    # Print overall classification report
    print("\nOverall Results (All Iterations):")
    print(classification_report(all_y_true, all_y_pred))

    # Save results to a text file
    with open(os.path.join(result_dir, "overall_results.txt"), "w") as f:
        f.write(f"Classifier: {CLASSICAL_CLASSIFIER_MODEL}\n")
        f.write(f"Use PCA: {NUM_PCA_COMPONENTS is not None}\n")
        f.write(f"PCA Components: {NUM_PCA_COMPONENTS}\n")
        f.write(f"Tune Hyperparameters: {tune_hyperparams}\n")
        f.write(f"Number of Folds: {NUM_KFOLDS}\n")
        f.write(f"Number of Iterations: {len(fold_features['iterations'])}\n")
        f.write(f"Use Data Preprocessing: {USE_DATA_PREPROCESSING}\n")
        if USE_DATA_PREPROCESSING:
            f.write(f"Preprocessing Method: {CLASSIFIER_APPROACH}\n\n")

        f.write("Overall Classification Report (All Iterations):\n")
        f.write(classification_report(all_y_true, all_y_pred))

        f.write("\nConfusion Matrix (All Iterations):\n")
        f.write(str(confusion_matrix(all_y_true, all_y_pred)))

    return {
        'fold_results': all_results['fold_results'],
        'best_model_info': best_model_metrics,
        'best_hyperparameters': best_model_metrics['hyperparameters'],
        'result_dir': result_dir
    }


def run_kfold_cross_validation(all_features, all_labels,
                               classifier_name=CLASSICAL_CLASSIFIER_MODEL,
                               use_pca=True, n_components=NUM_PCA_COMPONENTS,
                               tune_hyperparams=False, result_dir=None):
    """
    Run K-fold cross-validation for a classical ML model.

    Args:
        all_features (numpy.array): Feature matrix.
        all_labels (numpy.array): Target labels.
        classifier_name (str): Name of the classifier.
        use_pca (bool): Whether to use PCA for dimensionality reduction.
        n_components (int, optional): Number of PCA components.
        tune_hyperparams (bool): Whether to tune hyperparameters.
        result_dir (str, optional): Directory to save results.

    Returns:
        list: List of evaluation results for each fold.
    """
    # Initialize KFold
    from sklearn.model_selection import StratifiedKFold
    from config import NUM_KFOLDS, NUM_ITERATIONS

    all_iterations_results = {
        'fold_results': [],
        'all_y_true': [],
        'all_y_pred': []
    }

    # Informações do melhor modelo
    best_model_metrics = {
        'iteration': 0,
        'fold': 0,
        'accuracy': 0,
        'macro_avg_f1': 0,
        'model_path': None,
        'hyperparameters': None
    }

    # Verifica se o diretório do classificador existe
    if not result_dir:
        print("Erro: Diretório de resultados não especificado.")
        return None

    # Múltiplas iterações
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'=' * 50}")
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'=' * 50}")

        # Cria diretório para a iteração
        iter_dir = os.path.join(result_dir, f"iteration_{iteration + 1}")
        os.makedirs(iter_dir, exist_ok=True)
        os.makedirs(os.path.join(iter_dir, "plots"), exist_ok=True)

        # Inicializa StratifiedKFold
        skf = StratifiedKFold(n_splits=NUM_KFOLDS, shuffle=True, random_state=42 + iteration)

        # Armazena resultados desta iteração
        fold_results = []
        iteration_y_true = []
        iteration_y_pred = []

        # Executa cada fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_features, all_labels)):
            print(f"\n{'=' * 40}")
            print(f"Iteration {iteration + 1}, Fold {fold + 1}/{NUM_KFOLDS}")
            print(f"{'=' * 40}")

            # Divide dados
            train_features, val_features = all_features[train_idx], all_features[val_idx]
            train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

            # Cria diretório para o fold
            fold_dir = os.path.join(iter_dir, f"fold_{fold + 1}")
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(os.path.join(fold_dir, "plots"), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, "features"), exist_ok=True)

            # Caminho para salvar modelo
            model_save_path = os.path.join(fold_dir, "models", f"{classifier_name.lower()}_model.joblib")

            train_features = train_features.astype(np.float32)
            val_features = val_features.astype(np.float32)

            # Create model save path
            if iter_dir:
                model_save_path = os.path.join(
                    iter_dir,
                    "models",
                    f"{classifier_name.lower()}_iter_{iteration + 1}_fold_{fold + 1}.joblib"
                )
            else:
                model_save_path = None

            # Create fold result directory
            if iter_dir:
                fold_dir = os.path.join(iter_dir, f"fold_{fold + 1}")
                os.makedirs(fold_dir, exist_ok=True)
                os.makedirs(os.path.join(fold_dir, "plots"), exist_ok=True)
            else:
                fold_dir = None

            # Handle class imbalance based on the approach specified in config
            class_weights = None
            if USE_DATA_PREPROCESSING:
                print(f"Applying data preprocessing to iteration {iteration + 1}, fold {fold + 1} training data...")

                if CLASSIFIER_APPROACH == "class_weight":
                    train_features, train_labels, class_weights = apply_data_preprocessing(
                        features=train_features,
                        labels=train_labels,
                        method="class_weight",
                        random_state=42 + iteration * NUM_KFOLDS + fold
                        # Different random state for each fold and iteration
                    )
                elif CLASSIFIER_APPROACH == "hybrid":
                    train_features, train_labels, _ = apply_data_preprocessing(
                        features=train_features,
                        labels=train_labels,
                        method="hybrid",
                        random_state=42 + iteration * NUM_KFOLDS + fold
                    )
                elif CLASSIFIER_APPROACH == "undersampling":
                    train_features, train_labels, _ = apply_data_preprocessing(
                        features=train_features,
                        labels=train_labels,
                        method="undersampling",
                        random_state=42 + iteration * NUM_KFOLDS + fold
                    )
                else:  # Default to original SMOTE approach
                    train_features, train_labels, _ = apply_data_preprocessing(
                        features=train_features,
                        labels=train_labels,
                        method="smote",
                        random_state=42 + iteration * NUM_KFOLDS + fold
                    )

            # Create ML pipeline
            pipeline = create_ml_pipeline(
                classifier_name=classifier_name,
                use_pca=use_pca,
                n_components=n_components
            )

            try:
                # Train model
                if tune_hyperparams:
                    print(f"Tuning hyperparameters for iteration {iteration + 1}, fold {fold + 1}...")
                    param_grid = get_default_param_grid(classifier_name)

                    # Add class weight options for tree-based models
                    if class_weights is not None and classifier_name in ["RandomForest", "ExtraTrees", "AdaBoost"]:
                        if "classifier__class_weight" not in param_grid:
                            param_grid["classifier__class_weight"] = ['balanced', None]

                    grid_search = tune_hyperparameters(
                        pipeline=pipeline,
                        X=train_features,
                        y=train_labels,
                        param_grid=param_grid,
                        cv=3,  # Smaller CV for speed
                        class_weights=class_weights
                    )

                    # Use best model
                    fold_model = grid_search.best_estimator_
                else:
                    # Regular training with or without class weights
                    if class_weights is not None:
                        # Get the classifier from the pipeline
                        if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                            classifier = pipeline.named_steps['classifier']

                            # Train the preprocessing steps
                            if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
                                for name, transform in pipeline.steps[:-1]:  # All steps except classifier
                                    if hasattr(transform, "fit"):
                                        train_features = transform.fit_transform(train_features, train_labels)

                            # Check if classifier supports sample weights
                            if hasattr(classifier, 'fit') and 'sample_weight' in classifier.fit.__code__.co_varnames:
                                # Create sample weights from class weights
                                sample_weights = np.array([class_weights[label] for label in train_labels])
                                classifier.fit(train_features, train_labels, sample_weight=sample_weights)
                                print(
                                    f"Iteration {iteration + 1}, Fold {fold + 1}: Training with sample weights based on class imbalance")
                            else:
                                # If sample_weight not supported, try to set class_weight attribute
                                if hasattr(classifier, 'class_weight'):
                                    classifier.class_weight = class_weights
                                    print(f"Iteration {iteration + 1}, Fold {fold + 1}: Setting class_weight parameter")
                                classifier.fit(train_features, train_labels)

                            fold_model = pipeline
                        else:
                            # Fall back to regular pipeline fit
                            pipeline.fit(train_features, train_labels)
                            fold_model = pipeline
                    else:
                        # Train model with default parameters
                        pipeline.fit(train_features, train_labels)
                        fold_model = pipeline

                # Save model if path is provided
                if model_save_path:
                    save_model(fold_model, model_save_path)

                # Evaluate on validation set
                val_pred = fold_model.predict(val_features)

                # Add to iteration predictions
                iteration_y_true.extend(val_labels)
                iteration_y_pred.extend(val_pred)

                # Calculate metrics
                report = classification_report(val_labels, val_pred, output_dict=True)

                # Print classification report
                print(f"\nIteration {iteration + 1}, Fold {fold + 1} Validation Set Classification Report:")
                print(classification_report(val_labels, val_pred))

                if report['accuracy'] > best_model_metrics['accuracy'] or \
                        (report['accuracy'] == best_model_metrics['accuracy'] and
                         report['macro avg']['f1-score'] > best_model_metrics['macro_avg_f1']):

                    # For classical models, capture best hyperparameters
                    if hasattr(fold_model, 'get_params'):
                        model_params = fold_model.get_params()
                    else:
                        model_params = {}

                    # Store core configuration
                    hyperparameters = {
                        'classifier_name': classifier_name,
                        'use_pca': use_pca,
                        'n_components': n_components,
                        'class_weights': class_weights,
                        'preprocessing_approach': CLASSIFIER_APPROACH if USE_DATA_PREPROCESSING else None,
                        'model_params': model_params
                    }

                    best_model_metrics = {
                        'iteration': iteration + 1,
                        'fold': fold + 1,
                        'accuracy': report['accuracy'],
                        'macro_avg_f1': report['macro avg']['f1-score'],
                        'model_path': model_save_path,
                        'model': fold_model,  # Keep reference to the model object itself
                        'hyperparameters': hyperparameters
                    }

                # Store fold results
                fold_result = {
                    "iteration": iteration + 1,
                    "fold": fold + 1,
                    "accuracy": report["accuracy"],
                    "macro_avg_precision": report["macro avg"]["precision"],
                    "macro_avg_recall": report["macro avg"]["recall"],
                    "macro_avg_f1": report["macro avg"]["f1-score"],
                    "class_report": report
                }

                fold_results.append(fold_result)

            except Exception as e:
                print(f"Error in iteration {iteration + 1}, fold {fold + 1}: {e}")
                continue

        # Convert iteration predictions to arrays
        iteration_y_true = np.array(iteration_y_true)
        iteration_y_pred = np.array(iteration_y_pred)

        # Print overall classification report for this iteration
        print(f"\nOverall Iteration {iteration + 1} Results:")
        print(classification_report(iteration_y_true, iteration_y_pred))

        # Store iteration results
        all_iterations_results['fold_results'].extend(fold_results)
        all_iterations_results['all_y_true'].extend(iteration_y_true)
        all_iterations_results['all_y_pred'].extend(iteration_y_pred)

        # Calculate average fold metrics for this iteration
        avg_accuracy = np.mean([res["accuracy"] for res in fold_results])
        avg_precision = np.mean([res["macro_avg_precision"] for res in fold_results])
        avg_recall = np.mean([res["macro_avg_recall"] for res in fold_results])
        avg_f1 = np.mean([res["macro_avg_f1"] for res in fold_results])

        print(f"\nIteration {iteration + 1} Average Metrics:")
        print(f"Accuracy: {avg_accuracy:.4f}")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
        print(f"F1 Score: {avg_f1:.4f}")

    # Calculate overall metrics across all iterations
    all_y_true = np.array(all_iterations_results['all_y_true'])
    all_y_pred = np.array(all_iterations_results['all_y_pred'])

    # Print overall classification report
    print("\nOverall Results (All Iterations):")
    print(classification_report(all_y_true, all_y_pred))

    # Calculate average metrics across all iterations
    iteration_metrics = []
    for iteration in range(NUM_ITERATIONS):
        iter_results = [res for res in all_iterations_results['fold_results'] if res['iteration'] == iteration + 1]
        avg_accuracy = np.mean([res['accuracy'] for res in iter_results])
        avg_precision = np.mean([res['macro_avg_precision'] for res in iter_results])
        avg_recall = np.mean([res['macro_avg_recall'] for res in iter_results])
        avg_f1 = np.mean([res['macro_avg_f1'] for res in iter_results])

        iteration_metrics.append({
            'iteration': iteration + 1,
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        })

    # Overall average across all iterations
    overall_avg_accuracy = np.mean([m['accuracy'] for m in iteration_metrics])
    overall_avg_precision = np.mean([m['precision'] for m in iteration_metrics])
    overall_avg_recall = np.mean([m['recall'] for m in iteration_metrics])
    overall_avg_f1 = np.mean([m['f1'] for m in iteration_metrics])

    print(f"\nAverage Metrics Across All Iterations:")
    print(f"Accuracy: {overall_avg_accuracy:.4f}")
    print(f"Precision: {overall_avg_precision:.4f}")
    print(f"Recall: {overall_avg_recall:.4f}")
    print(f"F1 Score: {overall_avg_f1:.4f}")

    # Save results to a text file
    if result_dir:
        with open(os.path.join(result_dir, "overall_results.txt"), "w") as f:
            f.write(f"Classifier: {classifier_name}\n")
            f.write(f"Use PCA: {use_pca}\n")
            f.write(f"PCA Components: {n_components}\n")
            f.write(f"Tune Hyperparameters: {tune_hyperparams}\n")
            f.write(f"Number of Folds: {NUM_KFOLDS}\n")
            f.write(f"Number of Iterations: {NUM_ITERATIONS}\n")
            f.write(f"Use Data Preprocessing: {USE_DATA_PREPROCESSING}\n")
            if USE_DATA_PREPROCESSING:
                f.write(f"Preprocessing Method: {CLASSIFIER_APPROACH}\n\n")

            f.write("Average Metrics Across All Iterations:\n")
            f.write(f"Accuracy: {overall_avg_accuracy:.4f}\n")
            f.write(f"Precision: {overall_avg_precision:.4f}\n")
            f.write(f"Recall: {overall_avg_recall:.4f}\n")
            f.write(f"F1 Score: {overall_avg_f1:.4f}\n\n")

            f.write("Per-Iteration Metrics:\n")
            for m in iteration_metrics:
                f.write(f"Iteration {m['iteration']}:\n")
                f.write(f"  Accuracy: {m['accuracy']:.4f}\n")
                f.write(f"  Precision: {m['precision']:.4f}\n")
                f.write(f"  Recall: {m['recall']:.4f}\n")
                f.write(f"  F1 Score: {m['f1']:.4f}\n\n")

            f.write("Overall Classification Report (All Iterations):\n")
            f.write(classification_report(all_y_true, all_y_pred))

            f.write("\nConfusion Matrix (All Iterations):\n")
            f.write(str(confusion_matrix(all_y_true, all_y_pred)))

    return {
        'fold_results': all_iterations_results['fold_results'],
        'best_model_info': best_model_metrics,
        'best_model': best_model_metrics.get('model', None),
        'best_hyperparameters': best_model_metrics['hyperparameters'],
        'result_dir': result_dir
    }


def train_final_feature_extraction_model(all_features, all_labels, best_hyperparameters,
                                         result_dir, class_names=None, feature_extractor=None,
                                         all_paths=None, all_labels_orig=None, preprocess_fn=None):
    """
    Train a final classical ML model on all training data using the best hyperparameters.
    Can either use pre-extracted features or extract features on demand.

    Args:
        all_features: Combined training and validation features (can be None if paths provided)
        all_labels: Combined training and validation labels (can be None if paths provided)
        best_hyperparameters: Best hyperparameters found during cross-validation
        result_dir: Directory to save results
        class_names: List of class names
        feature_extractor: Feature extractor model (required if all_features is None)
        all_paths: Combined training and validation paths (required if all_features is None)
        all_labels_orig: Original labels for paths (required if all_features is None)
        preprocess_fn: Image preprocessing function

    Returns:
        Final trained model and evaluation results
    """
    print("\n" + "=" * 60)
    print("Training Final Classical ML Model on All Training Data")
    print("=" * 60)

    # Create final model directory
    final_model_dir = os.path.join(result_dir, CLASSICAL_CLASSIFIER_MODEL.lower(), "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    os.makedirs(os.path.join(final_model_dir, "plots"), exist_ok=True)

    # Final model save path
    final_model_path = os.path.join(final_model_dir, "final_ml_model.joblib")

    # Extract hyperparameters
    classifier_name = best_hyperparameters['classifier_name']
    use_pca = best_hyperparameters['use_pca']
    n_components = best_hyperparameters['n_components']
    initial_class_weights = best_hyperparameters.get('class_weights', None)

    # If features are not provided, extract them
    if all_features is None:
        if feature_extractor is None or all_paths is None or all_labels_orig is None:
            raise ValueError("Must provide either all_features and all_labels OR "
                             "feature_extractor, all_paths, and all_labels_orig")

        # Path for combined features
        combined_features_path = os.path.join(
            result_dir,
            "features",
            "training_features.npz"
        )

        # Extract features from all data
        print("Extracting features for final model training...")
        all_features, all_labels = load_or_extract_features(
            feature_extractor=feature_extractor,
            paths=all_paths,
            labels=all_labels_orig,
            preprocess_fn=preprocess_fn,
            model_name=CNN_MODEL,
            features_save_path=combined_features_path
        )

    # Ensure features are the right type
    all_features = all_features.astype(np.float32)

    # Apply data preprocessing if it was used in the best model
    class_weights = initial_class_weights
    if best_hyperparameters['preprocessing_approach'] is not None:
        print(f"Applying {best_hyperparameters['preprocessing_approach']} preprocessing to all training data...")
        preprocessed_features, preprocessed_labels, new_class_weights = apply_data_preprocessing(
            features=all_features,
            labels=all_labels,
            method=best_hyperparameters['preprocessing_approach'],
            random_state=42
        )

        if best_hyperparameters['preprocessing_approach'] == "class_weight":
            class_weights = new_class_weights
    else:
        preprocessed_features = all_features
        preprocessed_labels = all_labels

    # Create ML pipeline with best configuration
    pipeline = create_ml_pipeline(
        classifier_name=classifier_name,
        use_pca=use_pca,
        n_components=n_components
    )

    # If we have the model_params, try to set them directly
    if 'model_params' in best_hyperparameters and best_hyperparameters['model_params']:
        try:
            # Set parameters for the classifier in the pipeline
            if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                classifier = pipeline.named_steps['classifier']
                relevant_params = {k.replace('classifier__', ''): v for k, v in
                                   best_hyperparameters['model_params'].items()
                                   if k.startswith('classifier__')}
                classifier.set_params(**relevant_params)
        except Exception as e:
            print(f"Warning: Could not set all model parameters: {e}")

    # Train the model on all data
    print(f"Training final model on all {len(preprocessed_features)} samples...")

    # Handle class weights if they were used
    if class_weights is not None:
        # Check if the classifier supports sample weights
        if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
            classifier = pipeline.named_steps['classifier']

            # Train the preprocessing steps
            if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
                for name, transform in pipeline.steps[:-1]:  # All steps except classifier
                    if hasattr(transform, "fit"):
                        preprocessed_features = transform.fit_transform(preprocessed_features, preprocessed_labels)

            # Check if classifier supports sample weights
            if hasattr(classifier, 'fit') and 'sample_weight' in classifier.fit.__code__.co_varnames:
                # Create sample weights from class weights
                sample_weights = np.array([class_weights[label] for label in preprocessed_labels])
                classifier.fit(preprocessed_features, preprocessed_labels, sample_weight=sample_weights)
                print("Training with sample weights based on class imbalance")
            else:
                # If sample_weight not supported, try to set class_weight attribute
                if hasattr(classifier, 'class_weight'):
                    classifier.class_weight = class_weights
                    print("Setting class_weight parameter")
                classifier.fit(preprocessed_features, preprocessed_labels)

            final_model = pipeline
        else:
            # Fall back to regular pipeline fit
            pipeline.fit(preprocessed_features, preprocessed_labels)
            final_model = pipeline
    else:
        # Train normally
        pipeline.fit(preprocessed_features, preprocessed_labels)
        final_model = pipeline

    # Save the final model
    save_model(final_model, final_model_path)
    print(f"Final model trained and saved to: {final_model_path}")

    # If possible, plot feature importance
    if hasattr(final_model, 'named_steps') and 'classifier' in final_model.named_steps:
        classifier = final_model.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            fi_plot_path = os.path.join(final_model_dir, "plots", "final_model_feature_importance.png")

            # Get feature names or indices
            if hasattr(final_model, 'named_steps') and 'pca' in final_model.named_steps:
                # For PCA, feature names are PCA components
                feature_names = [f"PC{i + 1}" for i in range(len(classifier.feature_importances_))]
            else:
                # Otherwise, use feature indices
                feature_names = [f"Feature {i + 1}" for i in range(len(classifier.feature_importances_))]

    return final_model, final_model_dir


def train_multiple_final_models(all_features, all_labels, best_hyperparameters,
                                result_dir, class_names=None, feature_extractor=None,
                                all_paths=None, all_labels_orig=None, preprocess_fn=None,
                                num_models=10):
    """
    Train multiple final classical ML models on all training data using the best hyperparameters.
    This allows for statistical testing of model performance.

    Args:
        all_features: Combined training and validation features (can be None if paths provided)
        all_labels: Combined training and validation labels (can be None if paths provided)
        best_hyperparameters: Best hyperparameters found during cross-validation
        result_dir: Directory to save results
        class_names: List of class names
        feature_extractor: Feature extractor model (required if all_features is None)
        all_paths: Combined training and validation paths (required if all_features is None)
        all_labels_orig: Original labels for paths (required if all_features is None)
        preprocess_fn: Image preprocessing function
        num_models: Number of models to train (default: 10)

    Returns:
        List of trained models and their directories
    """
    print("\n" + "=" * 60)
    print(f"Training {num_models} Final Classical ML Models on All Training Data")
    print("=" * 60)

    # Create final models directory
    final_models_dir = os.path.join(result_dir, CLASSICAL_CLASSIFIER_MODEL.lower(), "final_models")
    os.makedirs(final_models_dir, exist_ok=True)

    # Extract hyperparameters
    classifier_name = best_hyperparameters['classifier_name']
    use_pca = best_hyperparameters['use_pca']
    n_components = best_hyperparameters['n_components']
    initial_class_weights = best_hyperparameters.get('class_weights', None)

    # If features are not provided, extract them
    if all_features is None:
        if feature_extractor is None or all_paths is None or all_labels_orig is None:
            raise ValueError("Must provide either all_features and all_labels OR "
                             "feature_extractor, all_paths, and all_labels_orig")

        # Path for combined features
        combined_features_path = os.path.join(
            result_dir,
            "features",
            "training_features.npz"
        )

        # Extract features from all data
        print("Extracting features for final model training...")
        all_features, all_labels = load_or_extract_features(
            feature_extractor=feature_extractor,
            paths=all_paths,
            labels=all_labels_orig,
            preprocess_fn=preprocess_fn,
            model_name=CNN_MODEL,
            features_save_path=combined_features_path,
            apply_augmentation=USE_FEATURE_AUGMENTATION
        )

    # Ensure features are the right type
    all_features = all_features.astype(np.float32)

    trained_models = []
    model_results = []

    # Train multiple models
    for model_idx in range(num_models):
        print(f"\n{'=' * 50}")
        print(f"Training Model {model_idx + 1}/{num_models}")
        print(f"{'=' * 50}")

        # Create model-specific directory
        model_dir = os.path.join(final_models_dir, f"model_{model_idx + 1}")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True)

        # Model save path
        model_path = os.path.join(model_dir, "final_ml_model.joblib")

        # Apply data preprocessing if it was used in the best model
        class_weights = initial_class_weights
        if best_hyperparameters['preprocessing_approach'] is not None:
            print(f"Applying {best_hyperparameters['preprocessing_approach']} preprocessing to all training data...")

            # Use different random state for each model
            random_state = 42 + model_idx

            preprocessed_features, preprocessed_labels, new_class_weights = apply_data_preprocessing(
                features=all_features,
                labels=all_labels,
                method=best_hyperparameters['preprocessing_approach'],
                random_state=random_state
            )

            if best_hyperparameters['preprocessing_approach'] == "class_weight":
                class_weights = new_class_weights
        else:
            preprocessed_features = all_features
            preprocessed_labels = all_labels

        # Create ML pipeline with best configuration
        pipeline = create_ml_pipeline(
            classifier_name=classifier_name,
            use_pca=use_pca,
            n_components=n_components,
            random_state=42 + model_idx  # Different random state for each model
        )

        # If we have the model_params, try to set them directly
        if 'model_params' in best_hyperparameters and best_hyperparameters['model_params']:
            try:
                # Set parameters for the classifier in the pipeline
                if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                    classifier = pipeline.named_steps['classifier']
                    relevant_params = {k.replace('classifier__', ''): v for k, v in
                                       best_hyperparameters['model_params'].items()
                                       if k.startswith('classifier__')}
                    # Update random state for the classifier
                    relevant_params['random_state'] = 42 + model_idx
                    classifier.set_params(**relevant_params)
            except Exception as e:
                print(f"Warning: Could not set all model parameters: {e}")

        # Train the model on all data
        print(f"Training model {model_idx + 1} on all {len(preprocessed_features)} samples...")

        # Handle class weights if they were used
        if class_weights is not None:
            # Check if the classifier supports sample weights
            if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                classifier = pipeline.named_steps['classifier']

                # Train the preprocessing steps
                if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
                    for name, transform in pipeline.steps[:-1]:  # All steps except classifier
                        if hasattr(transform, "fit"):
                            preprocessed_features = transform.fit_transform(preprocessed_features, preprocessed_labels)

                # Check if classifier supports sample weights
                if hasattr(classifier, 'fit') and 'sample_weight' in classifier.fit.__code__.co_varnames:
                    # Create sample weights from class weights
                    sample_weights = np.array([class_weights[label] for label in preprocessed_labels])
                    classifier.fit(preprocessed_features, preprocessed_labels, sample_weight=sample_weights)
                    print("Training with sample weights based on class imbalance")
                else:
                    # If sample_weight not supported, try to set class_weight attribute
                    if hasattr(classifier, 'class_weight'):
                        classifier.class_weight = class_weights
                        print("Setting class_weight parameter")
                    classifier.fit(preprocessed_features, preprocessed_labels)

                final_model = pipeline
            else:
                # Fall back to regular pipeline fit
                pipeline.fit(preprocessed_features, preprocessed_labels)
                final_model = pipeline
        else:
            # Train normally
            pipeline.fit(preprocessed_features, preprocessed_labels)
            final_model = pipeline

        # Save the model
        save_model(final_model, model_path)
        print(f"Model {model_idx + 1} trained and saved to: {model_path}")

        # If possible, plot feature importance
        if hasattr(final_model, 'named_steps') and 'classifier' in final_model.named_steps:
            classifier = final_model.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                fi_plot_path = os.path.join(model_dir, "plots", "feature_importance.png")

                # Get feature names or indices
                if hasattr(final_model, 'named_steps') and 'pca' in final_model.named_steps:
                    # For PCA, feature names are PCA components
                    feature_names = [f"PC{i + 1}" for i in range(len(classifier.feature_importances_))]
                else:
                    # Otherwise, use feature indices
                    feature_names = [f"Feature {i + 1}" for i in range(len(classifier.feature_importances_))]


        trained_models.append({
            'model': final_model,
            'model_path': model_path,
            'model_dir': model_dir,
            'model_idx': model_idx + 1
        })

    print(f"\nAll {num_models} models trained successfully!")

    return trained_models, final_models_dir


def evaluate_multiple_final_models(trained_models, test_features, test_labels,
                                   result_dir, class_names=None):
    """
    Evaluate multiple final models on the test set and perform statistical analysis.

    Args:
        trained_models: List of trained model dictionaries
        test_features: Test feature matrix
        test_labels: Test labels
        result_dir: Directory to save results
        class_names: List of class names

    Returns:
        Dictionary with evaluation results and statistical analysis
    """
    from scipy import stats
    import pandas as pd

    print("\n" + "=" * 60)
    print("Evaluating Multiple Final Models on Test Set")
    print("=" * 60)

    # Results storage
    all_results = {
        'model_metrics': [],
        'predictions': [],
        'accuracies': [],
        'f1_scores': [],
        'precisions': [],
        'recalls': [],
        'class_metrics': []  # NEW: Store per-class metrics
    }

    # Evaluate each model
    for model_info in trained_models:
        model = model_info['model']
        model_idx = model_info['model_idx']
        model_dir = model_info['model_dir']

        print(f"\nEvaluating Model {model_idx}/{len(trained_models)}...")

        # Get predictions
        test_pred = model.predict(test_features)

        # Calculate metrics
        test_report = classification_report(test_labels, test_pred, output_dict=True)

        # Store predictions and metrics
        all_results['predictions'].append(test_pred)
        all_results['accuracies'].append(test_report['accuracy'])
        all_results['f1_scores'].append(test_report['macro avg']['f1-score'])
        all_results['precisions'].append(test_report['macro avg']['precision'])
        all_results['recalls'].append(test_report['macro avg']['recall'])

        # Store detailed metrics
        all_results['model_metrics'].append({
            'model_idx': model_idx,
            'accuracy': test_report['accuracy'],
            'macro_avg_precision': test_report['macro avg']['precision'],
            'macro_avg_recall': test_report['macro avg']['recall'],
            'macro_avg_f1': test_report['macro avg']['f1-score'],
            'class_report': test_report
        })

        # Store per-class metrics
        for class_idx in range(len(class_names) if class_names else len(np.unique(test_labels))):
            class_key = str(class_idx)
            if class_key in test_report:
                class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
                all_results['class_metrics'].append({
                    'model_idx': model_idx,
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'precision': test_report[class_key]['precision'],
                    'recall': test_report[class_key]['recall'],
                    'f1_score': test_report[class_key]['f1-score'],
                    'support': test_report[class_key]['support']
                })

        # Save individual model results
        with open(os.path.join(model_dir, "test_results.txt"), "w") as f:
            f.write(f"Model {model_idx} Test Results\n")
            f.write(f"{'=' * 30}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(test_labels, test_pred))
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(test_labels, test_pred)))

    # Statistical Analysis
    print("\n" + "=" * 50)
    print("Statistical Analysis of Model Performance")
    print("=" * 50)

    # Convert to numpy arrays for easier computation
    accuracies = np.array(all_results['accuracies'])
    f1_scores = np.array(all_results['f1_scores'])
    precisions = np.array(all_results['precisions'])
    recalls = np.array(all_results['recalls'])

    # Calculate statistics
    stats_results = {
        'accuracy': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'median': np.median(accuracies)
        },
        'f1_score': {
            'mean': np.mean(f1_scores),
            'std': np.std(f1_scores),
            'min': np.min(f1_scores),
            'max': np.max(f1_scores),
            'median': np.median(f1_scores)
        },
        'precision': {
            'mean': np.mean(precisions),
            'std': np.std(precisions),
            'min': np.min(precisions),
            'max': np.max(precisions),
            'median': np.median(precisions)
        },
        'recall': {
            'mean': np.mean(recalls),
            'std': np.std(recalls),
            'min': np.min(recalls),
            'max': np.max(recalls),
            'median': np.median(recalls)
        }
    }

    # Calculate 95% confidence intervals
    for metric_name, values in [('accuracy', accuracies), ('f1_score', f1_scores),
                                ('precision', precisions), ('recall', recalls)]:
        ci = stats.t.interval(0.95, len(values) - 1, loc=np.mean(values),
                              scale=stats.sem(values))
        stats_results[metric_name]['95_ci'] = ci

    # Save statistical results
    stats_path = os.path.join(result_dir, "statistical_analysis.txt")
    with open(stats_path, "w") as f:
        f.write("Statistical Analysis of Multiple Final Models\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of models: {len(trained_models)}\n\n")

        for metric_name, metric_stats in stats_results.items():
            f.write(f"{metric_name.upper()}:\n")
            f.write(f"  Mean ± Std: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}\n")
            f.write(f"  Median: {metric_stats['median']:.4f}\n")
            f.write(f"  Min/Max: {metric_stats['min']:.4f} / {metric_stats['max']:.4f}\n")
            f.write(f"  95% CI: [{metric_stats['95_ci'][0]:.4f}, {metric_stats['95_ci'][1]:.4f}]\n\n")

    # Create a summary DataFrame
    df_results = pd.DataFrame(all_results['model_metrics'])
    df_summary = df_results[['model_idx', 'accuracy', 'macro_avg_precision',
                             'macro_avg_recall', 'macro_avg_f1']]

    # Save as CSV
    csv_path = os.path.join(result_dir, "model_performance_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    # NEW: Save per-class metrics
    if all_results['class_metrics']:
        df_class_metrics = pd.DataFrame(all_results['class_metrics'])
        class_csv_path = os.path.join(result_dir, "per_class_metrics.csv")
        df_class_metrics.to_csv(class_csv_path, index=False)

        # Calculate per-class statistics
        class_stats = {}
        for class_idx in df_class_metrics['class_idx'].unique():
            class_data = df_class_metrics[df_class_metrics['class_idx'] == class_idx]
            class_name = class_data['class_name'].iloc[0]

            class_stats[class_name] = {
                'f1_score': {
                    'mean': class_data['f1_score'].mean(),
                    'std': class_data['f1_score'].std(),
                    'min': class_data['f1_score'].min(),
                    'max': class_data['f1_score'].max()
                },
                'precision': {
                    'mean': class_data['precision'].mean(),
                    'std': class_data['precision'].std()
                },
                'recall': {
                    'mean': class_data['recall'].mean(),
                    'std': class_data['recall'].std()
                }
            }

    # Plot box plots of metrics
    raw_metrics = {
        'accuracy': accuracies,
        'f1_score': f1_scores,
        'precision': precisions,
        'recall': recalls
    }

    print("\nStatistical Summary:")
    print(f"Accuracy: {stats_results['accuracy']['mean']:.4f} ± {stats_results['accuracy']['std']:.4f}")
    print(f"F1-Score: {stats_results['f1_score']['mean']:.4f} ± {stats_results['f1_score']['std']:.4f}")
    print(
        f"95% CI for Accuracy: [{stats_results['accuracy']['95_ci'][0]:.4f}, {stats_results['accuracy']['95_ci'][1]:.4f}]")
    print(
        f"95% CI for F1-Score: [{stats_results['f1_score']['95_ci'][0]:.4f}, {stats_results['f1_score']['95_ci'][1]:.4f}]")

    # Return comprehensive results
    return {
        'all_results': all_results,
        'statistics': stats_results,
        'summary_df': df_summary,
        'class_metrics_df': df_class_metrics if all_results['class_metrics'] else None,
        'class_statistics': class_stats if all_results['class_metrics'] else None
    }


def run_feature_extraction_pipeline(train_files_path, val_files_path, test_files_path,
                                    use_kfold=False, fine_tune_extractor=True,
                                    balance_features=True, tune_hyperparams=True,
                                    class_names=None):
    """
    Run the complete feature extraction + classical ML pipeline.

    Args:
        train_files_path (str): Path to training files list.
        val_files_path (str): Path to validation files list.
        test_files_path (str): Path to test files list.
        use_kfold (bool): Whether to run K-fold cross-validation.
        fine_tune_extractor (bool): Whether to fine-tune the feature extractor.
        balance_features (bool): Whether to balance class representation in features.
        tune_hyperparams (bool): Whether to tune hyperparameters for classical models.
        class_names (list, optional): List of class names.

    Returns:
        dict: Results of the pipeline.
    """
    # Set up GPU memory
    setup_gpu_memory()

    # Create result directories
    dirs = create_feature_extraction_directories(
        base_dir=RESULTS_DIR,
        cnn_model_name=CNN_MODEL,
        classifier_name=CLASSICAL_CLASSIFIER_MODEL,
        num_iterations=NUM_ITERATIONS,
        num_folds=NUM_KFOLDS
    )
    print(f"Results will be saved to: {dirs['base']}")

    # Load data paths and labels
    train_paths, train_labels = load_paths_labels(train_files_path)
    val_paths, val_labels = load_paths_labels(val_files_path)
    test_paths, test_labels = load_paths_labels(test_files_path)

    # Create preprocessing function
    preprocess_fn = None
    if USE_GRAPHIC_PREPROCESSING:
        preprocess_fn = lambda img: apply_graphic_preprocessing(
            img,
            use_hair_removal=USE_HAIR_REMOVAL,
            use_contrast_enhancement=USE_ENHANCED_CONTRAST,
            use_segmentation=USE_IMAGE_SEGMENTATION,
            visualize=VISUALIZE
        )

    extractor_path = dirs['extractor']

    # Get feature extractor model
    feature_extractor, from_pretrained = get_feature_extractor_from_cnn(extractor_path)

    if use_kfold:
        # Combina treino e validação para validação cruzada
        all_paths = np.concatenate([train_paths, val_paths])
        all_labels = np.concatenate([train_labels, val_labels])

        print(f"Running k-fold cross-validation with {NUM_KFOLDS} folds...")

        # Extrai features separadamente para cada fold para evitar vazamento de dados
        fold_features = run_kfold_feature_extraction(
            all_paths=all_paths,
            all_labels=all_labels,
            dirs=dirs,
            class_names=class_names
        )

        # Treina modelos clássicos para cada fold usando as features extraídas
        cv_results = run_model_training_by_fold(
            fold_features=fold_features,
            tune_hyperparams=tune_hyperparams,
            result_dir=dirs['classifiers'][CLASSICAL_CLASSIFIER_MODEL]['base'],
            class_names=class_names
        )

        # Save detailed fold results
        save_fold_results(
            fold_results=cv_results['fold_results'],
            result_dir=dirs['classifiers'][CLASSICAL_CLASSIFIER_MODEL]['base'],
            classifier_name=CLASSICAL_CLASSIFIER_MODEL
        )

        # Train multiple final models with all training data using best hyperparameters
        try:
            print("\nStarting training of multiple final models...")

            trained_models, final_models_dir = train_multiple_final_models(
                all_features=None,
                all_labels=None,
                best_hyperparameters=cv_results['best_hyperparameters'],
                result_dir=dirs['base'],
                class_names=class_names,
                feature_extractor=feature_extractor,
                all_paths=all_paths,
                all_labels_orig=all_labels,
                preprocess_fn=preprocess_fn,
                num_models=NUM_FINAL_MODELS
            )
            print(f"All {NUM_FINAL_MODELS} final models trained and saved in: {final_models_dir}")

        except Exception as e:
            import traceback
            print(f"ERROR in training multiple final models: {e}")
            traceback.print_exc()
            trained_models = []
            final_models_dir = None

        # Extract test features for evaluation
        test_features_path = dirs['test_features']

        test_features, test_labels_out = load_or_extract_features(
            feature_extractor=feature_extractor,
            paths=test_paths,
            labels=test_labels,
            preprocess_fn=preprocess_fn,
            model_name=CNN_MODEL,
            features_save_path=test_features_path,
            apply_augmentation=False
        )

        test_features = test_features.astype(np.float32)

        # Evaluate all final models on test set
        if trained_models:
            print("\nEvaluating multiple final models on test set...")

            eval_results = evaluate_multiple_final_models(
                trained_models=trained_models,
                test_features=test_features,
                test_labels=test_labels_out,
                result_dir=final_models_dir,
                class_names=class_names
            )

            # Store comprehensive results
            results = {
                'k_fold': cv_results['fold_results'],
                'best_model_info': cv_results['best_model_info'],
                'final_models': trained_models,
                'final_models_evaluation': eval_results,
                'statistical_analysis': eval_results['statistics'],
                'fold_statistics': cv_results.get('fold_statistics', None)
            }

            # Save comprehensive summary
            summary_path = os.path.join(dirs['base'], "complete_experiment_summary.txt")
            with open(summary_path, "w") as f:
                f.write("Complete Experiment Summary\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Feature Extractor: {CNN_MODEL}\n")
                f.write(f"Classifier: {CLASSICAL_CLASSIFIER_MODEL}\n")
                f.write(f"Use CNN Data Augmentation: {USE_DATA_AUGMENTATION}\n")
                f.write(f"Use Feature Augmentation: {USE_FEATURE_AUGMENTATION}\n")
                f.write(f"Use Fine-tuning: {USE_FINE_TUNING}\n")
                f.write(f"Use Preprocessing: {USE_GRAPHIC_PREPROCESSING}\n")
                f.write(f"Use PCA: {NUM_PCA_COMPONENTS is not None}\n")
                f.write(f"PCA Components: {NUM_PCA_COMPONENTS}\n")
                f.write(f"Use Data Preprocessing: {USE_DATA_PREPROCESSING}\n")
                if USE_DATA_PREPROCESSING:
                    f.write(f"Preprocessing Method: {CLASSIFIER_APPROACH}\n")
                f.write(f"\nCross-validation:\n")
                f.write(f"  Iterations: {NUM_ITERATIONS}\n")
                f.write(f"  Folds per iteration: {NUM_KFOLDS}\n")
                f.write(f"  Total folds evaluated: {len(cv_results['fold_results'])}\n")
                f.write(f"\nFinal Models:\n")
                f.write(f"  Number of final models: {NUM_FINAL_MODELS}\n")
                f.write(f"  Models trained: {len(trained_models)}\n")

                if 'statistical_analysis' in results:
                    f.write(f"\nFinal Models Test Performance (Mean ± Std):\n")
                    stats = results['statistical_analysis']
                    f.write(f"  Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}\n")
                    f.write(f"  F1-Score: {stats['f1_score']['mean']:.4f} ± {stats['f1_score']['std']:.4f}\n")
                    f.write(f"  Precision: {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}\n")
                    f.write(f"  Recall: {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}\n")
                    f.write(f"\n95% Confidence Intervals:\n")
                    f.write(f"  Accuracy: [{stats['accuracy']['95_ci'][0]:.4f}, {stats['accuracy']['95_ci'][1]:.4f}]\n")
                    f.write(f"  F1-Score: [{stats['f1_score']['95_ci'][0]:.4f}, {stats['f1_score']['95_ci'][1]:.4f}]\n")

        else:
            print("No final models were trained. Falling back to best model from cross-validation...")

            # Fall back to evaluating the best model from cross-validation
            best_model_path = cv_results['best_model_info']['model_path']
            if os.path.exists(best_model_path):
                from models.classical_models import load_model
                best_model = load_model(best_model_path)

                # Evaluate on test set
                test_pred = best_model.predict(test_features)
                test_report = classification_report(test_labels_out, test_pred, output_dict=True)

                print("\nTest Set Classification Report (Best CV Model):")
                print(classification_report(test_labels_out, test_pred))

                results = {
                    'k_fold': cv_results['fold_results'],
                    'best_model_info': cv_results['best_model_info'],
                    'test_results': {
                        "accuracy": test_report["accuracy"],
                        "macro_avg_precision": test_report["macro avg"]["precision"],
                        "macro_avg_recall": test_report["macro avg"]["recall"],
                        "macro_avg_f1": test_report["macro avg"]["f1-score"],
                        "class_report": test_report
                    }
                }
    else:
        # Extração direta de features sem validação cruzada
        print("Extracting features without cross-validation...")

        # Paths para salvar features
        train_features_path = os.path.join(dirs['features'], "train_features.npz")
        val_features_path = os.path.join(dirs['features'], "val_features.npz")
        test_features_path = dirs['test_features']

        classifier_path = os.path.join(
            dirs['classifiers'][CLASSICAL_CLASSIFIER_MODEL]['models'],
            f"{CLASSICAL_CLASSIFIER_MODEL.lower()}_classifier.joblib"
        )

        # Extract features
        train_features, train_labels_out = load_or_extract_features(
            feature_extractor=feature_extractor,
            paths=train_paths,
            labels=train_labels,
            preprocess_fn=preprocess_fn,
            model_name=CNN_MODEL,
            features_save_path=train_features_path
        )

        val_features, val_labels_out = load_or_extract_features(
            feature_extractor=feature_extractor,
            paths=val_paths,
            labels=val_labels,
            preprocess_fn=preprocess_fn,
            model_name=CNN_MODEL,
            features_save_path=val_features_path
        )

        # Converte para float32 se necessário
        train_features = train_features.astype(np.float32)
        val_features = val_features.astype(np.float32)

        # Aplica pré-processamento de dados para lidar com desbalanceamento de classes
        preprocessing_info = {}
        if USE_DATA_PREPROCESSING:
            print("Applying data preprocessing to training features only...")
            print(f"Original training class distribution: {np.bincount(train_labels_out)}")

            # Aplica a abordagem selecionada com base em CLASSIFIER_APPROACH
            if CLASSIFIER_APPROACH == "class_weight":
                train_features, train_labels_out, class_weights = apply_data_preprocessing(
                    features=train_features,
                    labels=train_labels_out,
                    method="class_weight",
                    random_state=42
                )
                preprocessing_info = {
                    "method": "class_weight",
                    "class_weights": class_weights
                }
            elif CLASSIFIER_APPROACH == "hybrid":
                train_features, train_labels_out, _ = apply_data_preprocessing(
                    features=train_features,
                    labels=train_labels_out,
                    method="hybrid",
                    random_state=42
                )
                preprocessing_info = {
                    "method": "hybrid"
                }
            elif CLASSIFIER_APPROACH == "undersampling":
                train_features, train_labels_out, _ = apply_data_preprocessing(
                    features=train_features,
                    labels=train_labels_out,
                    method="undersampling",
                    random_state=42
                )
                preprocessing_info = {
                    "method": "undersampling"
                }
            else:  # Default para SMOTE
                train_features, train_labels_out, _ = apply_data_preprocessing(
                    features=train_features,
                    labels=train_labels_out,
                    method="smote",
                    random_state=42
                )
                preprocessing_info = {
                    "method": "smote"
                }

            if CLASSIFIER_APPROACH != "class_weight":
                print(f"After {CLASSIFIER_APPROACH}. New class distribution: {np.bincount(train_labels_out)}")
            print("Note: Validation and test data remain with original distribution")

        # Aplica balanceamento simples se solicitado (e não houver pré-processamento)
        elif balance_features and not USE_DATA_PREPROCESSING:
            print("Balancing features across classes...")
            train_features, train_labels_out = augment_features_with_balanced_sampling(
                features=train_features,
                labels=train_labels_out
            )
            print(f"After balancing. New class distribution: {np.bincount(train_labels_out)}")

        # Treina modelo único
        model, train_results = train_and_evaluate_classical_model(
            train_features=train_features,
            train_labels=train_labels_out,
            val_features=val_features,
            val_labels=val_labels_out,
            classifier_name=CLASSICAL_CLASSIFIER_MODEL,
            use_pca=(NUM_PCA_COMPONENTS is not None),
            n_components=NUM_PCA_COMPONENTS,
            tune_hyperparams=tune_hyperparams,
            result_dir=dirs['classifiers'][CLASSICAL_CLASSIFIER_MODEL]['base'],
            model_save_path=classifier_path,
            class_weights=preprocessing_info.get("class_weights", None)
        )

        # Extrai features de teste e avalia
        test_features, test_labels_out = load_or_extract_features(
            feature_extractor=feature_extractor,
            paths=test_paths,
            labels=test_labels,
            preprocess_fn=preprocess_fn,
            model_name=CNN_MODEL,
            features_save_path=test_features_path
        )

        test_features = test_features.astype(np.float32)

        # Avalia no conjunto de teste
        test_pred = model.predict(test_features)

        # Calcula métricas
        test_report = classification_report(test_labels_out, test_pred, output_dict=True)

        # Imprime relatório de classificação
        print("\nTest Set Classification Report:")
        print(classification_report(test_labels_out, test_pred))

        # Adicional: métricas ROC AUC
        if hasattr(model, 'predict_proba'):
            # Calcula ROC AUC para multi-classe
            if len(np.unique(test_labels_out)) > 2:
                # Abordagem One-vs-Rest para multi-classe
                y_test_bin = label_binarize(test_labels_out, classes=np.unique(test_labels_out))

                # Se o modelo suporta predict_proba
                y_score = model.predict_proba(test_features)

                # Calcula ROC AUC para cada classe
                roc_auc = {}
                for i in range(len(np.unique(test_labels_out))):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr, tpr)

                # Calcula ROC AUC médio (macro)
                macro_roc_auc = np.mean(list(roc_auc.values()))

                print(f"Macro-average ROC AUC: {macro_roc_auc:.4f}")

                # Adiciona aos resultados
                test_report["macro_roc_auc"] = macro_roc_auc
                test_report["class_roc_auc"] = roc_auc
            else:
                # Classificação binária
                y_score = model.predict_proba(test_features)[:, 1]
                roc_auc = roc_auc_score(test_labels_out, y_score)
                print(f"ROC AUC: {roc_auc:.4f}")
                # Adiciona aos resultados
                test_report["roc_auc"] = roc_auc

        # Salva resultados
        with open(os.path.join(dirs['classifiers'][CLASSICAL_CLASSIFIER_MODEL]['base'],
                               "test_results.txt"), "w") as f:
            f.write(f"Feature Extractor: {CNN_MODEL}\n")
            f.write(f"Classifier: {CLASSICAL_CLASSIFIER_MODEL}\n")
            f.write(f"Use Fine-tuning: {USE_FINE_TUNING}\n")
            f.write(f"Use Preprocessing: {USE_GRAPHIC_PREPROCESSING}\n")
            f.write(f"Use PCA: {NUM_PCA_COMPONENTS is not None}\n")
            f.write(f"PCA Components: {NUM_PCA_COMPONENTS}\n\n")
            f.write(f"Use Data Preprocessing: {USE_DATA_PREPROCESSING}\n")
            if USE_DATA_PREPROCESSING:
                f.write(f"Preprocessing Method: {CLASSIFIER_APPROACH}\n\n")

            f.write("Test Set Classification Report:\n")
            f.write(classification_report(test_labels_out, test_pred))

            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(test_labels_out, test_pred)))

            # Escreve ROC AUC se disponível
            if "macro_roc_auc" in test_report:
                f.write(f"\nMacro-average ROC AUC: {test_report['macro_roc_auc']:.4f}\n")
                f.write("Class-wise ROC AUC:\n")
                for class_idx, auc_value in test_report["class_roc_auc"].items():
                    class_name = class_names[class_idx] if class_names and class_idx < len(
                        class_names) else f"Class {class_idx}"
                    f.write(f"  {class_name}: {auc_value:.4f}\n")
            elif "roc_auc" in test_report:
                f.write(f"\nROC AUC: {test_report['roc_auc']:.4f}\n")

        # Armazena resultados
        test_results = {
            "accuracy": test_report["accuracy"],
            "macro_avg_precision": test_report["macro avg"]["precision"],
            "macro_avg_recall": test_report["macro avg"]["recall"],
            "macro_avg_f1": test_report["macro avg"]["f1-score"],
            "class_report": test_report
        }

        # Adiciona ROC AUC aos resultados se disponível
        if "macro_roc_auc" in test_report:
            test_results["macro_roc_auc"] = test_report["macro_roc_auc"]
            test_results["class_roc_auc"] = test_report["class_roc_auc"]
        elif "roc_auc" in test_report:
            test_results["roc_auc"] = test_report["roc_auc"]

        results = {
            'train_evaluation': train_results,
            'test_evaluation': test_results,
            'model': model
        }

    # Limpa memória
    clear_session()
    gc.collect()

    return results