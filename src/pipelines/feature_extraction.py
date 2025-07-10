"""
CNN feature extraction + classical ML pipeline.
Handles feature extraction, model training, and evaluation.
"""

import gc
import os
import sys
from typing import Optional, Any, List, Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.backend import clear_session

from utils.metadata_extractor import (
    extract_metadata_for_paths,
    MetadataFeatureExtractor,
    combine_cnn_and_metadata_features
)

sys.path.append('..')
from config import (
    BATCH_SIZE,
    USE_GRAPHIC_PREPROCESSING,
    USE_FEATURE_PREPROCESSING,
    USE_FINE_TUNING,
    NUM_KFOLDS,
    CNN_MODEL,
    CLASSICAL_CLASSIFIER_MODEL,
    RESULTS_DIR,
    USE_DATA_AUGMENTATION,
    IMG_SIZE,
    USE_HAIR_REMOVAL,
    USE_ENHANCED_CONTRAST,
    NUM_ITERATIONS,
    USE_FEATURE_AUGMENTATION,
    NUM_FINAL_MODELS,
    USE_METADATA, METADATA_PATH,
)

from utils.data_loaders import load_paths_labels, resize_image
from models.cnn_models import get_feature_extractor_model, get_feature_extractor_from_cnn
from models.classical_models import create_ml_pipeline, tune_hyperparameters, get_default_param_grid, save_model
from utils.fold_utils import save_fold_results
from preprocessing.feature.pipeline import apply_feature_preprocessing


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
    str_hair       = "hair_removal_" if USE_HAIR_REMOVAL else ""
    str_contrast   = "contrast_" if USE_ENHANCED_CONTRAST else ""
    str_graphic    = f"{str_contrast}{str_hair}" if USE_GRAPHIC_PREPROCESSING else ""
    str_augment    = "use_augmentation_" if USE_DATA_AUGMENTATION else ""
    str_preprocess = f"use_feature_preprocessing_" if USE_FEATURE_PREPROCESSING else ""
    str_feature    = f"use_feature_augmentation_" if USE_FEATURE_AUGMENTATION else ""
    str_meta       = "use_metadata_" if USE_METADATA else ""

    # Create main result directory path
    result_dir = os.path.join(base_dir,
                              f"feature_extraction_{cnn_model_name}_{str_graphic}{str_augment}{str_feature}{str_preprocess}{str_meta}")

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
        'final_models': os.path.join(classifier_dir, "final_models")
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

def extract_features_from_paths(feature_extractor, paths, labels=None, model_name=CNN_MODEL,
                                batch_size=BATCH_SIZE, apply_augmentation=USE_FEATURE_AUGMENTATION):
    """
    Extract features from image paths using a feature extractor model.
    Memory-efficient version to avoid GPU OOM errors.

    Args:
        feature_extractor: Feature extractor model.
        paths (numpy.array): Image paths.
        labels (numpy.array, optional): Image labels.
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
        from preprocessing.data.augmentation import AugmentationFactory
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

def load_cached(
    save_path: str
) -> Optional[Dict[str, Any]]:
    """
    Tenta carregar um arquivo .npz ou .npy e retorna um dict com:
      - features: np.ndarray
      - labels: Optional[np.ndarray]
      - metadata: dict (flags de pré-processamento e augmentation)
    Retorna None se o arquivo não existir ou for inválido.
    """
    if not save_path or not os.path.exists(save_path):
        return None

    try:
        if save_path.endswith(".npz"):
            data = np.load(save_path, allow_pickle=True)
            return {
                "features": data["features"].astype(np.float32),
                "labels": data.get("labels", None),
                "metadata": {
                    "augmentation_enabled": bool(data.get("augmentation_enabled", False)),
                    "metadata_enabled": bool(data.get("metadata_enabled", False)),
                },
                "augmentation_enabled": bool(data.get("augmentation_enabled", False)),
                "metadata_enabled": bool(data.get("metadata_enabled", False)),
            }

        elif save_path.endswith(".npy"):
            arr = np.load(save_path)
            return {
                "features": arr.astype(np.float32),
                "labels": None,
                "metadata": {"augmentation_enabled": None},
            }

    except Exception:
        # arquivo corrupto ou formato inesperado
        return None


def _handle_metadata_augmentation(
        metadata_features: np.ndarray,
        num_cnn_features: int,
        num_original_samples: int,
        apply_augmentation: bool
) -> np.ndarray:
    """
    Handle metadata features when CNN features are augmented.

    Args:
        metadata_features: Original metadata features
        num_cnn_features: Number of CNN features (possibly augmented)
        num_original_samples: Number of original samples
        apply_augmentation: Whether augmentation was applied

    Returns:
        Metadata features with same number of samples as CNN features
    """
    if not apply_augmentation or num_cnn_features == num_original_samples:
        # No augmentation or already matching
        return metadata_features

    # Calculate augmentation factor
    aug_factor = num_cnn_features // num_original_samples

    if aug_factor > 1:
        print(f"  → Replicating metadata features {aug_factor}x to match augmented CNN features")
        # Replicate metadata features for each augmented version
        metadata_features_aug = []
        for i in range(num_original_samples):
            # Repeat metadata for each augmented version of the image
            metadata_features_aug.extend([metadata_features[i]] * aug_factor)

        metadata_features = np.array(metadata_features_aug)
        print(f"  → Augmented metadata shape: {metadata_features.shape}")

    return metadata_features

def save_cache(
    save_path: str,
    features: np.ndarray,
    labels: Optional[np.ndarray],
    aug_enabled: bool,
    metadata_enabled: bool = False,
) -> None:
    """
    Salva num .npz (ou .npy) os arrays + metadados de preprocessing.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if save_path.endswith(".npz"):
        to_save = {
            "features": features,
            "augmentation_enabled": aug_enabled,
            "metadata_enabled": metadata_enabled,
        }
        if labels is not None:
            to_save["labels"] = labels
        np.savez(save_path, **to_save)
    else:
        np.save(save_path, features)

def load_or_extract_features(
    feature_extractor,
    paths: List[str],
    labels: Optional[np.ndarray] = None,
    model_name: Optional[str] = None,
    features_save_path: Optional[str] = None,
    apply_augmentation: bool = False,
    metadata_df: Optional[pd.DataFrame] = None,
    metadata_extractor: Optional[MetadataFeatureExtractor] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Carrega do cache ou extrai features, e aplica pré-processamento avançado se configurado.
    """
    # Extract metadata features for the original paths
    if metadata_df is not None and metadata_extractor is not None:
        metadata_features = extract_metadata_for_paths(
            image_paths=paths,
            metadata_df=metadata_df,
            metadata_extractor=metadata_extractor
        )
        print(f"  → metadata_features.shape = {metadata_features.shape}, expected {len(paths)} rows")
    else:
        metadata_features = None

    # 1) If no save path, extract and return
    if not features_save_path:
        print(f"Extracting {len(paths)} features (no cache)...")
        cnn_feats, labs = extract_features_from_paths(
            feature_extractor, paths, labels, model_name, BATCH_SIZE, apply_augmentation
        )

        # Handle augmentation for metadata features
        if metadata_features is not None:
            metadata_features = _handle_metadata_augmentation(
                metadata_features, cnn_feats.shape[0], len(paths), apply_augmentation
            )

        combined_features = combine_cnn_and_metadata_features(
            cnn_features=cnn_feats,
            metadata_features=metadata_features,
        )
        return combined_features, labs

    # 2) Try to load cache
    cache = load_cached(features_save_path)
    if cache:
        cnn_feats = cache["features"]
        labs = cache.get("labels") if cache.get("labels") is not None else labels

        # Check if cached features already include metadata
        if "metadata_enabled" in cache and cache["metadata_enabled"]:
            print("Using cached features with metadata")
            return cnn_feats, labs

        # If cache doesn't have metadata but we need it, combine
        if metadata_features is not None:
            print("Adding metadata to cached CNN features")

            # Handle augmentation for metadata features based on cached feature size
            metadata_features = _handle_metadata_augmentation(
                metadata_features, cnn_feats.shape[0], len(paths), apply_augmentation
            )

            combined_features = combine_cnn_and_metadata_features(
                cnn_features=cnn_feats,
                metadata_features=metadata_features,
            )

            # Update cache with combined features
            save_cache(
                features_save_path, combined_features, labs,
                apply_augmentation, metadata_enabled=True
            )

            return combined_features, labs
        else:
            print("Using cached CNN features without metadata")
            return cnn_feats, labs

    # 3) Cache not found - extract from scratch
    print(f"Extracting {len(paths)} features...")
    cnn_feats, out_labels = extract_features_from_paths(
        feature_extractor, paths, labels, model_name, BATCH_SIZE, apply_augmentation
    )

    if cnn_feats.size == 0:
        print("No features extracted!")
        return cnn_feats, out_labels

    # Handle augmentation for metadata features
    if metadata_features is not None:
        metadata_features = _handle_metadata_augmentation(
            metadata_features, cnn_feats.shape[0], len(paths), apply_augmentation
        )

    combined_features = combine_cnn_and_metadata_features(
        cnn_features=cnn_feats,
        metadata_features=metadata_features,
    )

    # Save combined features
    save_cache(
        features_save_path, combined_features, out_labels,
        apply_augmentation, metadata_enabled=(metadata_features is not None)
    )

    return combined_features, out_labels

def train_and_evaluate_classical_model(train_features, train_labels,
                                       val_features, val_labels,
                                       classifier_name=CLASSICAL_CLASSIFIER_MODEL,
                                       tune_hyperparams=True,
                                       model_save_path=None):
    """
    Train and evaluate a classical ML model on extracted features.

    Args:
        train_features (numpy.array): Training feature matrix.
        train_labels (numpy.array): Training labels.
        val_features (numpy.array): Validation feature matrix.
        val_labels (numpy.array): Validation labels.
        classifier_name (str): Name of the classifier.
        tune_hyperparams (bool): Whether to tune hyperparameters.
        model_save_path (str, optional): Path to save the trained model.

    Returns:
        tuple: (model, evaluation_results)
    """
    print(f"Training {classifier_name} classifier...")

    train_features = train_features.astype(np.float32)
    val_features = val_features.astype(np.float32)

    # Create ML pipeline
    pipeline = create_ml_pipeline(
        classifier_name=classifier_name
    )

    # Tune hyperparameters if requested
    if tune_hyperparams:
        print("Tuning hyperparameters...")
        param_grid = get_default_param_grid(classifier_name)

        grid_search = tune_hyperparameters(
            pipeline=pipeline,
            X=train_features,
            y=train_labels,
            param_grid=param_grid,
            cv=5
        )

        # Use best model
        model = grid_search.best_estimator_
    else:
        model = pipeline
        model.fit(train_features, train_labels)

    if model_save_path:
        save_model(model, model_save_path)

    val_pred = model.predict(val_features)

    report = classification_report(val_labels, val_pred, output_dict=True)

    print("\nValidation Set Classification Report:")
    print(classification_report(val_labels, val_pred))

    # Evaluation results
    evaluation_results = {
        "accuracy": report["accuracy"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "class_report": report
    }

    return model, evaluation_results


def run_kfold_feature_extraction(all_paths,
                                 all_labels,
                                 dirs,
                                 feature_extractor,
                                 metadata_df: Optional[pd.DataFrame] = None,
                                 metadata_extractor: Optional[MetadataFeatureExtractor] = None):
    """
    Run feature extraction for each fold of k-fold cross-validation.
    Extracts features separately for each fold to avoid data leakage.

    Args:
        all_paths (numpy.array): All image paths.
        all_labels (numpy.array): All labels.
        dirs (dict): Directory to save results.
        feature_extractor: FeatureExtractor object.
        metadata_df (Optional[pandas.DataFrame]): Pandas DataFrame containing metadata information.
        metadata_extractor (Optional[MetadataFeatureExtractor]): MetadataFeatureExtractor object.

    Returns:
        dict: Dictionary containing fold-specific features and paths
    """
    from sklearn.model_selection import StratifiedKFold
    from config import NUM_KFOLDS, NUM_ITERATIONS

    # Create feature extraction directory
    features_by_fold_dir = dirs['features_by_fold']
    os.makedirs(features_by_fold_dir, exist_ok=True)

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
                model_name=CNN_MODEL,
                features_save_path=train_features_path,
                apply_augmentation=USE_FEATURE_AUGMENTATION,
                metadata_extractor=metadata_extractor,
                metadata_df=metadata_df
            )

            val_features, val_labels_out = load_or_extract_features(
                feature_extractor=feature_extractor,
                paths=val_paths,
                labels=val_labels,
                model_name=CNN_MODEL,
                features_save_path=val_features_path,
                metadata_extractor=metadata_extractor,
                metadata_df=metadata_df
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


def run_model_training_by_fold(fold_features, result_dir, tune_hyperparams=True):
    """
    Train classical models using pre-extracted features for each fold.

    Args:
        fold_features (dict): Dictionary with fold-specific feature paths from run_kfold_feature_extraction
        result_dir (str): Directory to save results
        tune_hyperparams (bool): Whether to tune hyperparameters

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
            train_data     = np.load(train_features_path, allow_pickle=True)
            train_raw      = train_data['features'].astype(np.float32)
            train_labels   = train_data['labels']

            print(f"Loading validation features from: {val_features_path}")
            val_data       = np.load(val_features_path, allow_pickle=True)
            val_raw        = val_data['features'].astype(np.float32)
            val_labels     = val_data['labels']

            if USE_FEATURE_PREPROCESSING:
                # 1) fit on train, *and* get back the fitted pipeline
                proc_feat, proc_labels, feat_pipe = apply_feature_preprocessing(
                    features        = train_raw,
                    labels          = train_labels,
                    algorithm       = CLASSICAL_CLASSIFIER_MODEL,
                    training        = True,
                    save_path       = None
                )
                # 2) transform valid with the very same pipeline
                proc_val_feat, proc_val_lab = feat_pipe.transform(val_raw, val_labels)
            else:
                proc_feat, proc_labels, proc_val_feat, proc_val_lab = train_raw, train_labels, val_raw, val_labels

            # Print data shapes
            print(f"Training features shape: {proc_feat.shape}")
            print(f"Validation features shape: {proc_val_feat.shape}")
            print(f"Training class distribution: {np.bincount(proc_labels)}")
            print(f"Validation class distribution: {np.bincount(proc_val_lab)}")

            # Train model
            try:
                model, evaluation = train_and_evaluate_classical_model(
                    train_features=proc_feat,
                    train_labels=proc_labels,
                    val_features=proc_val_feat,
                    val_labels=proc_val_lab,
                    classifier_name=CLASSICAL_CLASSIFIER_MODEL,
                    tune_hyperparams=tune_hyperparams,
                    model_save_path=model_save_path
                )

                # Get predictions
                val_pred = model.predict(proc_val_feat)

                # Add to iteration predictions
                iteration_y_true.extend(proc_val_lab)
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
            del proc_feat, proc_val_feat, proc_labels, proc_val_lab, model
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
        f.write(f"Tune Hyperparameters: {tune_hyperparams}\n")
        f.write(f"Number of Folds: {NUM_KFOLDS}\n")
        f.write(f"Number of Iterations: {len(fold_features['iterations'])}\n")
        f.write(f"Use Feature Preprocessing: {USE_FEATURE_PREPROCESSING}\n")

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


def run_kfold_cross_validation(all_features,
                               all_labels,
                               classifier_name=CLASSICAL_CLASSIFIER_MODEL,
                               tune_hyperparams=False,
                               result_dir=None):
    """
    Run K-fold cross-validation for a classical ML model.

    Args:
        all_features (numpy.array): Feature matrix.
        all_labels (numpy.array): Target labels.
        classifier_name (str): Name of the classifier.
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

            # Create ML pipeline
            pipeline = create_ml_pipeline(
                classifier_name=classifier_name
            )

            try:
                # Train model
                if tune_hyperparams:
                    print(f"Tuning hyperparameters for iteration {iteration + 1}, fold {fold + 1}...")
                    param_grid = get_default_param_grid(classifier_name)
                    grid_search = tune_hyperparameters(
                        pipeline=pipeline,
                        X=train_features,
                        y=train_labels,
                        param_grid=param_grid,
                        cv=3,
                    )
                    fold_model = grid_search.best_estimator_
                else:
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
                        'model_params': model_params
                    }

                    best_model_metrics = {
                        'iteration': iteration + 1,
                        'fold': fold + 1,
                        'accuracy': report['accuracy'],
                        'macro_avg_f1': report['macro avg']['f1-score'],
                        'model_path': model_save_path,
                        'model': fold_model,
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
            f.write(f"Tune Hyperparameters: {tune_hyperparams}\n")
            f.write(f"Number of Folds: {NUM_KFOLDS}\n")
            f.write(f"Number of Iterations: {NUM_ITERATIONS}\n")
            f.write(f"Use Feature Preprocessing: {USE_FEATURE_PREPROCESSING}\n")

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

def train_multiple_final_models(all_features, all_labels, best_hyperparameters,
                                result_dir,feature_extractor=None,
                                all_paths=None, all_labels_orig=None,
                                num_models=10,
                                metadata_df: Optional[pd.DataFrame] = None,
                                metadata_extractor: Optional[MetadataFeatureExtractor] = None
                                ):
    """
    Train multiple final classical ML models on all training data using the best hyperparameters.
    This allows for statistical testing of model performance.

    Args:
        all_features: Combined training and validation features (can be None if paths provided)
        all_labels: Combined training and validation labels (can be None if paths provided)
        best_hyperparameters: Best hyperparameters found during cross-validation
        result_dir: Directory to save results
        feature_extractor: Feature extractor model (required if all_features is None)
        all_paths: Combined training and validation paths (required if all_features is None)
        all_labels_orig: Original labels for paths (required if all_features is None)
        num_models: Number of models to train (default: 10)
        metadata_df: Pandas DataFrame containing metadata (required if all_features is None)
        metadata_extractor: MetadataFeatureExtractor model (required if all_features is None)

    Returns:
        List of trained models and their directories
    """
    print("\n" + "=" * 60)
    print(f"Training {num_models} Final Classical ML Models on All Training Data")
    print("=" * 60)

    # Create final models directory
    final_models_dir = os.path.join(result_dir['base'], CLASSICAL_CLASSIFIER_MODEL.lower(), "final_models")
    os.makedirs(final_models_dir, exist_ok=True)

    # Extract hyperparameters
    classifier_name = best_hyperparameters['classifier_name']

    # If features are not provided, extract them
    if all_features is None:
        if feature_extractor is None or all_paths is None or all_labels_orig is None:
            raise ValueError("Must provide either all_features and all_labels OR "
                             "feature_extractor, all_paths, and all_labels_orig")

        # Path for combined features
        combined_features_path = os.path.join(
            result_dir['base'],
            "features",
            "training_features.npz"
        )

        # Extract features from all data
        print("Extracting features for final model training...")
        all_features, all_labels = load_or_extract_features(
            feature_extractor=feature_extractor,
            paths=all_paths,
            labels=all_labels_orig,
            model_name=CNN_MODEL,
            features_save_path=combined_features_path,
            apply_augmentation=USE_FEATURE_AUGMENTATION,
            metadata_extractor=metadata_extractor,
            metadata_df=metadata_df
        )

    # Ensure features are the right type
    all_features = all_features.astype(np.float32)

    if USE_FEATURE_PREPROCESSING:
        # fit the pipeline on the *entire* training set once…
        all_feat_proc, all_lab_proc, feat_pipe = apply_feature_preprocessing(
            features=all_features,
            labels=all_labels,
            algorithm=CLASSICAL_CLASSIFIER_MODEL,
            training=True,
            save_path=os.path.join(result_dir['classifiers'][CLASSICAL_CLASSIFIER_MODEL]['final_models'],  "final_feat_pipe.joblib")
        )
    else:
        all_feat_proc, all_lab_proc, feat_pipe = all_features, all_labels, None

    trained_models = []

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

        # Create ML pipeline with best configuration
        pipeline = create_ml_pipeline(
            classifier_name=classifier_name,
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
        print(f"Training model {model_idx + 1} on all {len(all_features)} samples...")

        pipeline.fit(all_feat_proc, all_lab_proc)
        final_model = pipeline

        # Save the model
        save_model(final_model, model_path)
        print(f"Model {model_idx + 1} trained and saved to: {model_path}")

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
                                    use_kfold=False,
                                    tune_hyperparams=True,
                                    class_names=None):
    """
    Run the complete feature extraction + classical ML pipeline.

    Args:
        train_files_path (str): Path to training files list.
        val_files_path (str): Path to validation files list.
        test_files_path (str): Path to test files list.
        use_kfold (bool): Whether to run K-fold cross-validation.
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

    extractor_path = dirs['extractor']

    # Get feature extractor model
    feature_extractor, from_pretrained = get_feature_extractor_from_cnn(extractor_path)

    if USE_METADATA:
        metadata_df = pd.read_csv(METADATA_PATH)
        metadata_extractor_path = os.path.join(RESULTS_DIR, 'metadata_extractor', 'metadata_extractor.joblib')
        if os.path.exists(metadata_extractor_path):
            print("Loading existing metadata extractor...")
            metadata_extractor = MetadataFeatureExtractor.load(metadata_extractor_path)
        else:
            print("Creating and fitting metadata extractor...")
            metadata_extractor = MetadataFeatureExtractor()
            metadata_extractor.fit(metadata_df)
            metadata_extractor.save(metadata_extractor_path)
    else:
        metadata_extractor = None
        metadata_df = None

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
            feature_extractor=feature_extractor,
            metadata_extractor=metadata_extractor,
            metadata_df=metadata_df
        )

        # Treina modelos clássicos para cada fold usando as features extraídas
        cv_results = run_model_training_by_fold(
            fold_features=fold_features,
            tune_hyperparams=tune_hyperparams,
            result_dir=dirs['classifiers'][CLASSICAL_CLASSIFIER_MODEL]['base']
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
                result_dir=dirs,
                feature_extractor=feature_extractor,
                all_paths=all_paths,
                all_labels_orig=all_labels,
                num_models=NUM_FINAL_MODELS,
                metadata_extractor=metadata_extractor,
                metadata_df=metadata_df
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
            model_name=CNN_MODEL,
            features_save_path=test_features_path,
            apply_augmentation=False,
            metadata_extractor=metadata_extractor,
            metadata_df=metadata_df
        )

        test_features = test_features.astype(np.float32)

        if USE_FEATURE_PREPROCESSING:
            proc_test_feat, proc_test_lab, pipe = apply_feature_preprocessing(
                features=test_features,
                labels=test_labels_out,
                algorithm=CLASSICAL_CLASSIFIER_MODEL,
                training=False,
                save_path=os.path.join(dirs['classifiers'][CLASSICAL_CLASSIFIER_MODEL]['final_models'], "final_feat_pipe.joblib")
            )
        else:
            proc_test_feat, proc_test_lab, pipe = test_features, test_labels_out, None

        if trained_models:
            print("\nEvaluating multiple final models on test set...")

            eval_results = evaluate_multiple_final_models(
                trained_models=trained_models,
                test_features=proc_test_feat,
                test_labels=proc_test_lab,
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
                f.write(f"Use Feature Preprocessing: {USE_FEATURE_PREPROCESSING}\n")
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

    # Limpa memória
    clear_session()
    gc.collect()

    return results