"""
CNN feature extraction + classical ML pipeline.
Handles feature extraction, model training, and evaluation.
"""

import datetime
import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.backend import clear_session

from skincancer.src.config import IMG_SIZE, USE_DATA_PREPROCESSING

sys.path.append('..')
from config import (
    BATCH_SIZE,
    USE_GRAPHIC_PREPROCESSING,
    USE_DATA_PREPROCESSING,
    USE_FINE_TUNING,
    NUM_KFOLDS,
    NUM_EPOCHS,
    CNN_MODEL,
    CLASSICAL_CLASSIFIER_MODEL,
    NUM_PCA_COMPONENTS,
    RESULTS_DIR,
    USE_DATA_AUGMENTATION,
    VISUALIZE
)

from utils.data_loaders import load_paths_labels, load_and_preprocess_dataset, resize_image
from utils.preprocessing import apply_graphic_preprocessing
from models.cnn_models import get_feature_extractor_model, fine_tune_feature_extractor
from models.classical_models import create_ml_pipeline, tune_hyperparameters, get_default_param_grid, save_model


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


def create_result_directories(base_dir=RESULTS_DIR):
    """
    Create directories for saving results.

    Args:
        base_dir (str): Base directory for results.

    Returns:
        str: Path to the created result directory.
    """
    timestamp      = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    str_graphic    = "use_graphic_preprocessing_" if USE_GRAPHIC_PREPROCESSING else ""
    str_augment    = "use_augmentation_" if USE_DATA_AUGMENTATION else ""
    str_preprocess = "use_data_preprocess_" if USE_DATA_PREPROCESSING else ""
    result_dir     = os.path.join(base_dir, f"feature_extraction_{str_graphic}{str_augment}{str_preprocess}{timestamp}")

    # Create subdirectories
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)

    return result_dir


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true (numpy.array): True labels.
        y_pred (numpy.array): Predicted labels.
        class_names (list, optional): List of class names.
        save_path (str, optional): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    # Get the number of classes from the confusion matrix
    num_classes = cm.shape[0]

    # If class_names is not provided, use numeric indices
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Now create the heatmap with the class names
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to: {save_path}")

    plt.close()


def plot_feature_importance(model, feature_names=None, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained ML model with feature_importances_ attribute.
        feature_names (list, optional): List of feature names.
        top_n (int): Number of top features to show.
        save_path (str, optional): Path to save the plot.
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute. Skipping feature importance plot.")
        return

    # Get feature importances
    importances = model.feature_importances_

    # Use indices as feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    # Limit to top_n features
    if top_n is not None and top_n < len(indices):
        indices = indices[:top_n]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {len(indices)} Feature Importances")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to: {save_path}")

    plt.close()


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


def extract_features_from_paths(feature_extractor, paths, labels=None,
                                preprocess_fn=None, model_name=CNN_MODEL,
                                batch_size=BATCH_SIZE, apply_augmentation=USE_DATA_AUGMENTATION):
    """
    Extract features from image paths using a feature extractor model.

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
    # Load and preprocess images
    print(f"Loading and preprocessing {len(paths)} images...")

    # Create a preprocessing function for each image
    def process_image(path, augment=False, aug_index=None):
        import cv2
        from skincancer.src.utils.data_loaders import apply_model_preprocessing

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

    # Process images in batches
    num_samples = len(paths)
    num_batches = int(np.ceil(num_samples / batch_size))

    all_features = []
    all_labels = []

    augmentation_pipelines = None
    if apply_augmentation:
        from utils.augmentation import AugmentationFactory
        augmentation_pipelines = AugmentationFactory.get_feature_extraction_augmentation()
        num_augmentations = len(augmentation_pipelines)
    else:
        num_augmentations = 1

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_paths = paths[start_idx:end_idx]

        # Process batch of images
        batch_images = []
        batch_indices = []

        for j, path in enumerate(batch_paths):
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

        if not batch_images:
            continue

        # Convert to numpy array
        batch_images = np.array(batch_images)

        # Extract features
        batch_features = feature_extractor.predict(batch_images, verbose=0)

        all_features.append(batch_features)

        # Add labels if provided
        if labels is not None:
            # Repeat each label for the original + augmented versions
            if apply_augmentation:
                batch_labels = np.repeat(labels[np.unique(batch_indices)], num_augmentations)
            else:
                batch_labels = labels[batch_indices]
            all_labels.append(batch_labels)

    # Concatenate results
    features = np.concatenate(all_features, axis=0) if all_features else np.array([])
    features = features.astype(np.float32)

    if labels is not None and all_labels:
        labels_out = np.concatenate(all_labels, axis=0)
        return features, labels_out

    return features


def load_or_extract_features(feature_extractor, paths, labels=None,
                             preprocess_fn=None, model_name=CNN_MODEL,
                             features_save_path=None):
    """
    Load cached features or extract new ones.

    Args:
        feature_extractor: Feature extractor model.
        paths (numpy.array): Image paths.
        labels (numpy.array, optional): Image labels.
        preprocess_fn (callable, optional): Function for image preprocessing.
        model_name (str): CNN model name for preprocessing.
        features_save_path (str, optional): Path to save/load features.

    Returns:
        tuple: (features, labels) if labels is provided, otherwise just features.
    """
    if features_save_path and os.path.exists(features_save_path):
        print(f"Loading cached features from: {features_save_path}")
        features_data = np.load(features_save_path, allow_pickle=True)

        if isinstance(features_data, np.ndarray) and len(features_data.shape) == 2:
            # Only features were saved
            features = features_data.astype(np.float32)
            if labels is not None:
                return features, labels
            return features
        elif isinstance(features_data, dict) and 'features' in features_data and 'labels' in features_data:
            # Both features and labels were saved
            features = features_data['features'].astype(np.float32)
            labels_loaded = features_data['labels']

            if labels is not None:
                # Verify labels match
                if np.array_equal(labels, labels_loaded):
                    return features, labels
                else:
                    print("Warning: Cached labels don't match provided labels. Re-extracting features.")
            else:
                return features, labels_loaded

    # Extract features
    print(f"Extracting features from {len(paths)} images...")

    if labels is not None:
        features, labels_out = extract_features_from_paths(
            feature_extractor=feature_extractor,
            paths=paths,
            labels=labels,
            preprocess_fn=preprocess_fn,
            model_name=model_name
        )
    else:
        features = extract_features_from_paths(
            feature_extractor=feature_extractor,
            paths=paths,
            preprocess_fn=preprocess_fn,
            model_name=model_name
        )
        labels_out = None

    # Save features if path is provided
    if features_save_path:
        print(f"Saving features to: {features_save_path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(features_save_path), exist_ok=True)

        features = features.astype(np.float32)

        if labels_out is not None:
            # Save both features and labels
            np.savez(features_save_path, features=features, labels=labels_out)
        else:
            # Save only features
            np.save(features_save_path, features)

    if labels is not None:
        return features, labels_out
    else:
        return features


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

            # Then add synthetic samples until reaching target count
            samples_needed = target_count - len(cls_indices)

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
                                       model_save_path=None):
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
        cm_plot_path = os.path.join(result_dir, "plots", "confusion_matrix.png")
        plot_confusion_matrix(val_labels, val_pred, save_path=cm_plot_path)

        # Plot feature importance if model supports it
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

                plot_feature_importance(classifier, feature_names, save_path=fi_plot_path)

    # Evaluation results
    evaluation_results = {
        "accuracy": report["accuracy"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "class_report": report
    }

    return model, evaluation_results


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
    kf = KFold(n_splits=NUM_KFOLDS, shuffle=True, random_state=42)

    # List to store evaluation results
    fold_results = []

    # Dictionary to collect predictions
    all_y_true = []
    all_y_pred = []

    # Run each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_features)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{NUM_KFOLDS}")
        print(f"{'=' * 50}")

        # Split data
        train_features, val_features = all_features[train_idx], all_features[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

        train_features = train_features.astype(np.float32)
        val_features = val_features.astype(np.float32)

        # Create model save path
        if result_dir:
            model_save_path = os.path.join(
                result_dir,
                "models",
                f"{classifier_name.lower()}_fold_{fold + 1}.joblib"
            )
        else:
            model_save_path = None

        # Create fold result directory
        if result_dir:
            fold_dir = os.path.join(result_dir, f"fold_{fold + 1}")
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(os.path.join(fold_dir, "plots"), exist_ok=True)
        else:
            fold_dir = None

        try:
            # Create ML pipeline
            pipeline = create_ml_pipeline(
                classifier_name=classifier_name,
                use_pca=use_pca,
                n_components=n_components
            )

            # Tune hyperparameters if requested
            if tune_hyperparams:
                print(f"Tuning hyperparameters for fold {fold + 1}...")
                param_grid = get_default_param_grid(classifier_name)

                grid_search = tune_hyperparameters(
                    pipeline=pipeline,
                    X=train_features,
                    y=train_labels,
                    param_grid=param_grid,
                    cv=3  # Smaller CV for speed
                )

                # Use best model
                model = grid_search.best_estimator_
            else:
                # Train model with default parameters
                model = pipeline
                model.fit(train_features, train_labels)

            # Save model if path is provided
            if model_save_path:
                save_model(model, model_save_path)

            # Evaluate on validation set
            val_pred = model.predict(val_features)

            # Add to overall predictions
            all_y_true.extend(val_labels)
            all_y_pred.extend(val_pred)

            # Calculate metrics
            report = classification_report(val_labels, val_pred, output_dict=True)

            # Print classification report
            print(f"\nFold {fold + 1} Validation Set Classification Report:")
            print(classification_report(val_labels, val_pred))

            # Plot confusion matrix if result directory is provided
            if fold_dir:
                cm_plot_path = os.path.join(fold_dir, "plots", "confusion_matrix.png")
                plot_confusion_matrix(val_labels, val_pred, save_path=cm_plot_path)

            # Store fold results
            fold_result = {
                "fold": fold + 1,
                "accuracy": report["accuracy"],
                "macro_avg_precision": report["macro avg"]["precision"],
                "macro_avg_recall": report["macro avg"]["recall"],
                "macro_avg_f1": report["macro avg"]["f1-score"],
                "class_report": report
            }

            fold_results.append(fold_result)

        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            continue

    # Calculate overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Print overall classification report
    print("\nOverall K-Fold Cross-Validation Results:")
    print(classification_report(all_y_true, all_y_pred))

    # Plot overall confusion matrix
    if result_dir:
        cm_plot_path = os.path.join(result_dir, "plots", "overall_confusion_matrix.png")
        plot_confusion_matrix(all_y_true, all_y_pred, save_path=cm_plot_path)

    # Calculate average fold metrics
    avg_accuracy = np.mean([res["accuracy"] for res in fold_results])
    avg_precision = np.mean([res["macro_avg_precision"] for res in fold_results])
    avg_recall = np.mean([res["macro_avg_recall"] for res in fold_results])
    avg_f1 = np.mean([res["macro_avg_f1"] for res in fold_results])

    print(f"\nAverage K-Fold Metrics:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    # Save results to a text file
    if result_dir:
        with open(os.path.join(result_dir, "kfold_results.txt"), "w") as f:
            f.write(f"Classifier: {classifier_name}\n")
            f.write(f"Use PCA: {use_pca}\n")
            f.write(f"PCA Components: {n_components}\n")
            f.write(f"Tune Hyperparameters: {tune_hyperparams}\n")
            f.write(f"Number of Folds: {NUM_KFOLDS}\n\n")

            f.write("Average K-Fold Metrics:\n")
            f.write(f"Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Precision: {avg_precision:.4f}\n")
            f.write(f"Recall: {avg_recall:.4f}\n")
            f.write(f"F1 Score: {avg_f1:.4f}\n\n")

            f.write("Overall Classification Report:\n")
            f.write(classification_report(all_y_true, all_y_pred))

            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(all_y_true, all_y_pred)))

            f.write("\n\nIndividual Fold Results:\n")
            for fold, result in enumerate(fold_results):
                f.write(f"\nFold {fold + 1}:\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"Precision: {result['macro_avg_precision']:.4f}\n")
                f.write(f"Recall: {result['macro_avg_recall']:.4f}\n")
                f.write(f"F1 Score: {result['macro_avg_f1']:.4f}\n")

    return fold_results


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
    result_dir = create_result_directories()
    print(f"Results will be saved to: {result_dir}")

    # Load data paths and labels
    train_paths, train_labels = load_paths_labels(train_files_path)
    val_paths, val_labels = load_paths_labels(val_files_path)
    test_paths, test_labels = load_paths_labels(test_files_path)

    # Create preprocessing function
    preprocess_fn = None
    if USE_GRAPHIC_PREPROCESSING:
        preprocess_fn = lambda img: apply_graphic_preprocessing(
            img,
            use_hair_removal=False,
            use_contrast_enhancement=False,
            use_segmentation=False,
            visualize=VISUALIZE
        )

    # Paths for saving models and features
    extractor_save_path = os.path.join(
        result_dir,
        "models",
        f"{CNN_MODEL.lower()}_feature_extractor.h5"
    )

    train_features_save_path = os.path.join(
        result_dir,
        "features",
        "train_features.npz"
    )

    val_features_save_path = os.path.join(
        result_dir,
        "features",
        "val_features.npz"
    )

    test_features_save_path = os.path.join(
        result_dir,
        "features",
        "test_features.npz"
    )

    classifier_save_path = os.path.join(
        result_dir,
        "models",
        f"{CLASSICAL_CLASSIFIER_MODEL.lower()}_classifier.joblib"
    )

    # Get feature extractor model
    feature_extractor, loaded = get_feature_extractor_model(
        model_name=CNN_MODEL,
        fine_tune=USE_FINE_TUNING,
        save_path=extractor_save_path
    )

    # Fine-tune feature extractor if requested
    if fine_tune_extractor and not loaded:
        print("Fine-tuning feature extractor...")

        # Load a small subset of data for fine-tuning
        # We'll use a maximum of 2000 samples to avoid memory issues
        max_samples = min(len(train_paths), 2000)
        subset_indices = np.random.choice(len(train_paths), max_samples, replace=False)

        subset_train_paths = train_paths[subset_indices]
        subset_train_labels = train_labels[subset_indices]

        # Load and preprocess images
        X_train, y_train = load_and_preprocess_dataset(
            paths=subset_train_paths,
            labels=subset_train_labels,
            model_name=CNN_MODEL,
            preprocess_fn=preprocess_fn
        )

        # Same for validation
        max_val_samples = min(len(val_paths), 500)
        subset_val_indices = np.random.choice(len(val_paths), max_val_samples, replace=False)

        subset_val_paths = val_paths[subset_val_indices]
        subset_val_labels = val_labels[subset_val_indices]

        X_val, y_val = load_and_preprocess_dataset(
            paths=subset_val_paths,
            labels=subset_val_labels,
            model_name=CNN_MODEL,
            preprocess_fn=preprocess_fn
        )

        # Fine-tune the feature extractor
        feature_extractor = fine_tune_feature_extractor(
            feature_extractor=feature_extractor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            save_path=extractor_save_path,
            use_augmentation=USE_DATA_AUGMENTATION
        )

    # Extract features
    print("Extracting features...")

    train_features, train_labels_out = load_or_extract_features(
        feature_extractor=feature_extractor,
        paths=train_paths,
        labels=train_labels,
        preprocess_fn=preprocess_fn,
        model_name=CNN_MODEL,
        features_save_path=train_features_save_path
    )

    val_features, val_labels_out = load_or_extract_features(
        feature_extractor=feature_extractor,
        paths=val_paths,
        labels=val_labels,
        preprocess_fn=preprocess_fn,
        model_name=CNN_MODEL,
        features_save_path=val_features_save_path
    )

    # Balance features if requested
    if balance_features:
        print("Balancing features across classes...")
        train_features, train_labels_out = augment_features_with_balanced_sampling(
            features=train_features,
            labels=train_labels_out
        )

    # Combine training and validation features for cross-validation
    if use_kfold:
        all_features = np.concatenate([train_features, val_features])
        all_labels = np.concatenate([train_labels_out, val_labels_out])

        # Run cross-validation
        fold_results = run_kfold_cross_validation(
            all_features=all_features,
            all_labels=all_labels,
            classifier_name=CLASSICAL_CLASSIFIER_MODEL,
            use_pca=(NUM_PCA_COMPONENTS is not None and USE_DATA_PREPROCESSING),
            n_components=NUM_PCA_COMPONENTS,
            tune_hyperparams=tune_hyperparams,
            result_dir=result_dir
        )

        # Train final model on all data
        final_model, _ = train_and_evaluate_classical_model(
            train_features=all_features,
            train_labels=all_labels,
            val_features=all_features,  # Use same data for quick validation
            val_labels=all_labels,
            classifier_name=CLASSICAL_CLASSIFIER_MODEL,
            use_pca=(NUM_PCA_COMPONENTS is not None and USE_DATA_PREPROCESSING),
            n_components=NUM_PCA_COMPONENTS,
            tune_hyperparams=False,  # Already tuned in cross-validation
            result_dir=result_dir,
            model_save_path=classifier_save_path
        )

        results = {
            'k_fold': fold_results,
            'final_model': final_model
        }
    else:
        # Train single model
        model, train_results = train_and_evaluate_classical_model(
            train_features=train_features,
            train_labels=train_labels_out,
            val_features=val_features,
            val_labels=val_labels_out,
            classifier_name=CLASSICAL_CLASSIFIER_MODEL,
            use_pca=(NUM_PCA_COMPONENTS is not None and USE_DATA_PREPROCESSING),
            n_components=NUM_PCA_COMPONENTS,
            tune_hyperparams=tune_hyperparams,
            result_dir=result_dir,
            model_save_path=classifier_save_path
        )

        # Extract test features and evaluate
        test_features, test_labels_out = load_or_extract_features(
            feature_extractor=feature_extractor,
            paths=test_paths,
            labels=test_labels,
            preprocess_fn=preprocess_fn,
            model_name=CNN_MODEL,
            features_save_path=test_features_save_path
        )

        test_features = test_features.astype(np.float32)

        # Evaluate on test set
        test_pred = model.predict(test_features)

        # Calculate metrics
        test_report = classification_report(test_labels_out, test_pred, output_dict=True)

        # Print classification report
        print("\nTest Set Classification Report:")
        print(classification_report(test_labels_out, test_pred))

        # Plot confusion matrix
        cm_plot_path = os.path.join(result_dir, "plots", "test_confusion_matrix.png")
        plot_confusion_matrix(test_labels_out, test_pred, class_names, cm_plot_path)

        # Save test results
        with open(os.path.join(result_dir, "test_results.txt"), "w") as f:
            f.write(f"Feature Extractor: {CNN_MODEL}\n")
            f.write(f"Classifier: {CLASSICAL_CLASSIFIER_MODEL}\n")
            f.write(f"Use Fine-tuning: {USE_FINE_TUNING}\n")
            f.write(f"Use Preprocessing: {USE_GRAPHIC_PREPROCESSING}\n")
            f.write(f"Use PCA: {NUM_PCA_COMPONENTS is not None}\n")
            f.write(f"PCA Components: {NUM_PCA_COMPONENTS}\n\n")

            f.write("Test Set Classification Report:\n")
            f.write(classification_report(test_labels_out, test_pred))

            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(test_labels_out, test_pred)))

        # Store results
        test_results = {
            "accuracy": test_report["accuracy"],
            "macro_avg_precision": test_report["macro avg"]["precision"],
            "macro_avg_recall": test_report["macro avg"]["recall"],
            "macro_avg_f1": test_report["macro avg"]["f1-score"],
            "class_report": test_report
        }

        results = {
            'train_evaluation': train_results,
            'test_evaluation': test_results,
            'model': model
        }

    # Clear memory
    clear_session()
    gc.collect()

    return results