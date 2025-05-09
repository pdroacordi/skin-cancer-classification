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
    NUM_EPOCHS,
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
    USE_ENHANCED_CONTRAST
)

from utils.data_loaders import load_paths_labels, load_and_preprocess_dataset, resize_image
from utils.graphic_preprocessing import apply_graphic_preprocessing
from models.cnn_models import get_feature_extractor_model, fine_tune_feature_extractor
from models.classical_models import create_ml_pipeline, tune_hyperparameters, get_default_param_grid, save_model
from utils.data_preprocessing import apply_data_preprocessing


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
    str_hair       = "hair_removal_" if USE_HAIR_REMOVAL else ""
    str_contrast   = "contrast_" if USE_ENHANCED_CONTRAST else ""
    str_segmented  = "segmentation_" if USE_IMAGE_SEGMENTATION else ""
    str_graphic    = f"{str_segmented}{str_contrast}{str_hair}" if USE_GRAPHIC_PREPROCESSING else ""
    str_augment    = "use_augmentation_" if USE_DATA_AUGMENTATION else ""
    str_preprocess = f"use_data_preprocess_{CLASSIFIER_APPROACH}" if USE_DATA_PREPROCESSING else ""
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


def plot_roc_curves(model, X, y, class_names=None, save_path=None):
    """
    Plot ROC curves for each class in a multi-class problem.

    Args:
        model: Trained classifier with predict_proba method.
        X (numpy.array): Feature matrix.
        y (numpy.array): Target labels.
        class_names (list, optional): List of class names.
        save_path (str, optional): Path to save the plot.
    """
    if not hasattr(model, 'predict_proba'):
        print("Model does not support predict_proba. Skipping ROC curve plot.")
        return

    # Get unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Use class names if provided
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_classes]
    elif len(class_names) < n_classes:
        class_names = [f'Class {i}' for i in unique_classes]

    # Binarize the labels for multi-class ROC
    y_bin = label_binarize(y, classes=unique_classes)

    # Predict probabilities
    y_score = model.predict_proba(X)

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(12, 8))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(
            fpr[i], tpr[i],
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
        )

    # Plot the random guess line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")

    # Save or show
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve plot saved to: {save_path}")
    else:
        plt.show()

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
    from sklearn.model_selection import StratifiedKFold
    from config import NUM_KFOLDS, NUM_ITERATIONS

    # Dictionary to store all iteration results
    all_iterations_results = {
        'fold_results': [],
        'all_y_true': [],
        'all_y_pred': []
    }

    best_model_metrics = {
        'iteration': 0,
        'fold': 0,
        'accuracy': 0,
        'macro_avg_f1': 0,
        'model_path': None,
        'hyperparameters': None
    }

    # Run multiple iterations
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'=' * 50}")
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'=' * 50}")

        # Create iteration directory
        if result_dir:
            iter_dir = os.path.join(result_dir, f"iteration_{iteration + 1}")
            os.makedirs(iter_dir, exist_ok=True)
            os.makedirs(os.path.join(iter_dir, "plots"), exist_ok=True)
        else:
            iter_dir = None

        # Initialize StratifiedKFold with a different random state for each iteration
        skf = StratifiedKFold(n_splits=NUM_KFOLDS, shuffle=True, random_state=42 + iteration)

        # List to store evaluation results for this iteration
        fold_results = []

        # Dictionary to collect predictions for this iteration
        iteration_y_true = []
        iteration_y_pred = []

        # Run each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_features, all_labels)):
            print(f"\n{'=' * 40}")
            print(f"Iteration {iteration + 1}, Fold {fold + 1}/{NUM_KFOLDS}")
            print(f"{'=' * 40}")

            # Split data
            train_features, val_features = all_features[train_idx], all_features[val_idx]
            train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

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

                # Plot confusion matrix if result directory is provided
                if fold_dir:
                    cm_plot_path = os.path.join(fold_dir, "plots", "confusion_matrix.png")
                    plot_confusion_matrix(val_labels, val_pred, save_path=cm_plot_path)

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

        # Plot overall confusion matrix for this iteration
        if iter_dir:
            cm_plot_path = os.path.join(iter_dir, "plots", "overall_confusion_matrix.png")
            plot_confusion_matrix(iteration_y_true, iteration_y_pred, save_path=cm_plot_path)

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

        # Save iteration results to a text file
        if iter_dir:
            with open(os.path.join(iter_dir, "iteration_results.txt"), "w") as f:
                f.write(f"Classifier: {classifier_name}\n")
                f.write(f"Use PCA: {use_pca}\n")
                f.write(f"PCA Components: {n_components}\n")
                f.write(f"Tune Hyperparameters: {tune_hyperparams}\n")
                f.write(f"Number of Folds: {NUM_KFOLDS}\n")
                f.write(f"Iteration: {iteration + 1}/{NUM_ITERATIONS}\n")
                f.write(f"Use Data Preprocessing: {USE_DATA_PREPROCESSING}\n")
                if USE_DATA_PREPROCESSING:
                    f.write(f"Preprocessing Method: {CLASSIFIER_APPROACH}\n\n")

                f.write(f"Iteration {iteration + 1} Average Metrics:\n")
                f.write(f"Accuracy: {avg_accuracy:.4f}\n")
                f.write(f"Precision: {avg_precision:.4f}\n")
                f.write(f"Recall: {avg_recall:.4f}\n")
                f.write(f"F1 Score: {avg_f1:.4f}\n\n")

                f.write("Classification Report:\n")
                f.write(classification_report(iteration_y_true, iteration_y_pred))

                f.write("\nConfusion Matrix:\n")
                f.write(str(confusion_matrix(iteration_y_true, iteration_y_pred)))

    # Calculate overall metrics across all iterations
    all_y_true = np.array(all_iterations_results['all_y_true'])
    all_y_pred = np.array(all_iterations_results['all_y_pred'])

    # Print overall classification report
    print("\nOverall Results (All Iterations):")
    print(classification_report(all_y_true, all_y_pred))

    # Plot overall confusion matrix
    if result_dir:
        cm_plot_path = os.path.join(result_dir, "plots", "overall_confusion_matrix.png")
        plot_confusion_matrix(all_y_true, all_y_pred, save_path=cm_plot_path)

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


def train_final_feature_extraction_model(all_features, all_labels, best_hyperparameters, result_dir, class_names=None):
    """
    Train a final classical ML model on all training data using the best hyperparameters.

    Args:
        all_features: Combined training and validation features
        all_labels: Combined training and validation labels
        best_hyperparameters: Best hyperparameters found during cross-validation
        result_dir: Directory to save results
        class_names: List of class names

    Returns:
        Final trained model and evaluation results
    """
    print("\n" + "=" * 60)
    print("Training Final Classical ML Model on All Training Data")
    print("=" * 60)

    # Create final model directory
    final_model_dir = os.path.join(result_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    os.makedirs(os.path.join(final_model_dir, "plots"), exist_ok=True)

    # Final model save path
    final_model_path = os.path.join(final_model_dir, "final_ml_model.joblib")

    # Extract hyperparameters
    classifier_name = best_hyperparameters['classifier_name']
    use_pca = best_hyperparameters['use_pca']
    n_components = best_hyperparameters['n_components']
    class_weights = best_hyperparameters.get('class_weights', None)

    # Apply data preprocessing if it was used in the best model
    preprocessed_features = all_features
    preprocessed_labels = all_labels

    if best_hyperparameters['preprocessing_approach'] is not None:
        print(f"Applying {best_hyperparameters['preprocessing_approach']} preprocessing to all training data...")
        preprocessed_features, preprocessed_labels, class_weights = apply_data_preprocessing(
            features=all_features,
            labels=all_labels,
            method=best_hyperparameters['preprocessing_approach'],
            random_state=42
        )

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

            plot_feature_importance(classifier, feature_names, save_path=fi_plot_path)

    return final_model, final_model_dir

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
            use_hair_removal=USE_HAIR_REMOVAL,
            use_contrast_enhancement=USE_ENHANCED_CONTRAST,
            use_segmentation=USE_IMAGE_SEGMENTATION,
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
        # We'll use a maximum of 3000 samples to avoid memory issues
        max_samples = min(len(train_paths), 3000)
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

    # Apply data preprocessing to handle class imbalance if enabled
    preprocessing_info = {}
    if USE_DATA_PREPROCESSING:
        print("Applying data preprocessing to training features only...")

        # Print class distribution before preprocessing
        print(f"Original training class distribution: {np.bincount(train_labels_out)}")

        # Apply the selected approach based on CLASSIFIER_APPROACH
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
        else:  # Default to SMOTE
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

    # Apply simple balancing if requested (and data preprocessing not enabled)
    elif balance_features:
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
        cv_results = run_kfold_cross_validation(
            all_features=all_features,
            all_labels=all_labels,
            classifier_name=CLASSICAL_CLASSIFIER_MODEL,
            use_pca=(NUM_PCA_COMPONENTS is not None),
            n_components=NUM_PCA_COMPONENTS,
            tune_hyperparams=tune_hyperparams,
            result_dir=result_dir
        )

        # Train final model with all training data using best hyperparameters
        final_model, final_model_dir = train_final_feature_extraction_model(
            all_features=all_features,
            all_labels=all_labels,
            best_hyperparameters=cv_results['best_hyperparameters'],
            result_dir=result_dir,
            class_names=class_names
        )

        # Extract test features and evaluate final model
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
        print("\nEvaluating final model on test set...")
        test_pred = final_model.predict(test_features)

        # Calculate metrics
        test_report = classification_report(test_labels_out, test_pred, output_dict=True)

        # Print classification report
        print("\nTest Set Classification Report (Final Model):")
        print(classification_report(test_labels_out, test_pred))

        # Plot confusion matrix
        cm_plot_path = os.path.join(final_model_dir, "plots", "final_model_test_confusion_matrix.png")
        plot_confusion_matrix(test_labels_out, test_pred, class_names, cm_plot_path)

        # Store test results
        test_results = {
            "accuracy": test_report["accuracy"],
            "macro_avg_precision": test_report["macro avg"]["precision"],
            "macro_avg_recall": test_report["macro avg"]["recall"],
            "macro_avg_f1": test_report["macro avg"]["f1-score"],
            "class_report": test_report
        }

        # Save results to a text file
        with open(os.path.join(final_model_dir, "final_model_test_results.txt"), "w") as f:
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

        results = {
            'k_fold': cv_results['fold_results'],
            'best_model_info': cv_results['best_model_info'],
            'final_model_test_results': test_results
        }
    else:
        # Train single model
        model, train_results = train_and_evaluate_classical_model(
            train_features=train_features,
            train_labels=train_labels_out,
            val_features=val_features,
            val_labels=val_labels_out,
            classifier_name=CLASSICAL_CLASSIFIER_MODEL,
            use_pca=(NUM_PCA_COMPONENTS is not None),
            n_components=NUM_PCA_COMPONENTS,
            tune_hyperparams=tune_hyperparams,
            result_dir=result_dir,
            model_save_path=classifier_save_path,
            class_weights=preprocessing_info.get("class_weights", None)
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

        # Additional ROC AUC metrics
        if hasattr(model, 'predict_proba'):
            # Calculate ROC AUC for multi-class
            if len(np.unique(test_labels_out)) > 2:
                # One-vs-Rest approach for multi-class
                y_test_bin = label_binarize(test_labels_out, classes=np.unique(test_labels_out))

                # If model supports predict_proba
                y_score = model.predict_proba(test_features)

                # Calculate ROC AUC for each class
                roc_auc = {}
                for i in range(len(np.unique(test_labels_out))):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr, tpr)

                # Calculate macro-average ROC AUC
                macro_roc_auc = np.mean(list(roc_auc.values()))

                print(f"Macro-average ROC AUC: {macro_roc_auc:.4f}")

                # Add to test results
                test_report["macro_roc_auc"] = macro_roc_auc
                test_report["class_roc_auc"] = roc_auc

                # Plot ROC curves if result directory is provided
                if result_dir:
                    roc_plot_path = os.path.join(result_dir, "plots", "roc_curves.png")
                    plot_roc_curves(model, test_features, test_labels_out, class_names, roc_plot_path)
            else:
                # Binary classification
                y_score = model.predict_proba(test_features)[:, 1]
                roc_auc = roc_auc_score(test_labels_out, y_score)

                print(f"ROC AUC: {roc_auc:.4f}")

                # Add to test results
                test_report["roc_auc"] = roc_auc

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
            f.write(f"Use Data Preprocessing: {USE_DATA_PREPROCESSING}\n")
            if USE_DATA_PREPROCESSING:
                f.write(f"Preprocessing Method: {CLASSIFIER_APPROACH}\n\n")

            f.write("Test Set Classification Report:\n")
            f.write(classification_report(test_labels_out, test_pred))

            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(test_labels_out, test_pred)))

            # Write ROC AUC if available
            if "macro_roc_auc" in test_report:
                f.write(f"\nMacro-average ROC AUC: {test_report['macro_roc_auc']:.4f}\n")
                f.write("Class-wise ROC AUC:\n")
                for class_idx, auc_value in test_report["class_roc_auc"].items():
                    class_name = class_names[class_idx] if class_names and class_idx < len(
                        class_names) else f"Class {class_idx}"
                    f.write(f"  {class_name}: {auc_value:.4f}\n")
            elif "roc_auc" in test_report:
                f.write(f"\nROC AUC: {test_report['roc_auc']:.4f}\n")

        # Store results
        test_results = {
            "accuracy": test_report["accuracy"],
            "macro_avg_precision": test_report["macro avg"]["precision"],
            "macro_avg_recall": test_report["macro avg"]["recall"],
            "macro_avg_f1": test_report["macro avg"]["f1-score"],
            "class_report": test_report
        }

        # Add ROC AUC to results if available
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

    # Clear memory
    clear_session()
    gc.collect()

    return results