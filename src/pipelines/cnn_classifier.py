"""
End-to-end CNN classification pipeline.
Handles training, evaluation, and K-fold cross-validation.
"""

import datetime
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.backend import clear_session

sys.path.append('..')
from config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    USE_GRAPHIC_PREPROCESSING,
    USE_DATA_AUGMENTATION,
    USE_FINE_TUNING,
    VISUALIZE,
    NUM_KFOLDS,
    CNN_MODEL,
    RESULTS_DIR,
    USE_DATA_PREPROCESSING
)

from utils.data_loaders import load_paths_labels, MemoryEfficientDataGenerator
from utils.preprocessing import apply_graphic_preprocessing
from utils.augmentation import AugmentationFactory
from models.cnn_models import load_or_create_cnn, get_callbacks, create_model_name


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
    """Create directories for saving results."""
    timestamp      = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    str_graphic    = "use_graphic_preprocessing_" if USE_GRAPHIC_PREPROCESSING else ""
    str_augment    = "use_augmentation_" if USE_DATA_AUGMENTATION else ""
    str_preprocess = "use_data_preprocess_" if USE_DATA_PREPROCESSING else ""
    result_dir     = os.path.join(base_dir, f"cnn_classifier_{str_graphic}{str_augment}{str_preprocess}{timestamp}")

    # Create subdirectories
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "logs"), exist_ok=True)

    return result_dir


def plot_training_history(history, save_path=None):
    """
    Plot training history metrics.

    Args:
        history: Keras history object.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(12, 8))

    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to: {save_path}")

    plt.close()


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


def run_single_fold_training(train_paths, train_labels, val_paths, val_labels,
                             model_save_path, result_dir):
    """
    Run a single training fold with a CNN classifier.

    Args:
        train_paths (numpy.array): Training image paths.
        train_labels (numpy.array): Training labels.
        val_paths (numpy.array): Validation image paths.
        val_labels (numpy.array): Validation labels.
        model_save_path (str): Path to save the model.
        result_dir (str): Directory to save results.

    Returns:
        tuple: (model, history)
    """
    # Get augmentation pipeline if enabled
    augment_fn = None
    if USE_DATA_AUGMENTATION:
        augmentation = AugmentationFactory.get_medium_augmentation()
        augment_fn = lambda img: augmentation(image=img)['image']

    # Preprocessing function
    preprocess_fn = None
    if USE_GRAPHIC_PREPROCESSING:
        preprocess_fn = lambda img: apply_graphic_preprocessing(
            img,
            use_hair_removal=True,
            use_contrast_enhancement=True,
            use_segmentation=False,
            visualize=VISUALIZE
        )

    # Create data generators
    train_gen = MemoryEfficientDataGenerator(
        paths=train_paths,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        model_name=CNN_MODEL,
        preprocess_fn=preprocess_fn,
        augment_fn=augment_fn,
        shuffle=True
    )

    val_gen = MemoryEfficientDataGenerator(
        paths=val_paths,
        labels=val_labels,
        batch_size=BATCH_SIZE,
        model_name=CNN_MODEL,
        preprocess_fn=preprocess_fn,
        augment_fn=None,  # No augmentation for validation
        shuffle=False
    )

    # Calculate steps per epoch
    steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
    validation_steps = math.ceil(len(val_paths) / BATCH_SIZE)

    # TensorBoard log directory
    log_dir = os.path.join(result_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Create or load model
    model, loaded = load_or_create_cnn(
        model_name=CNN_MODEL,
        mode='classifier',
        fine_tune=USE_FINE_TUNING,
        save_path=model_save_path
    )

    # Train model if not loaded
    if not loaded:
        print(f"Training CNN classifier...")

        # Get callbacks
        callbacks = get_callbacks(model_save_path, log_dir)

        # Train the model
        history = model.fit(
            train_gen.get_keras_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=NUM_EPOCHS,
            validation_data=val_gen.get_keras_generator(),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Plot training history
        history_plot_path = os.path.join(result_dir, "plots", "training_history.png")
        plot_training_history(history, history_plot_path)

        return model, history
    else:
        print(f"Using pre-trained model from: {model_save_path}")
        return model, None


def evaluate_model(model, test_paths, test_labels, result_dir, class_names=None):
    """
    Evaluate a trained CNN model on test data.

    Args:
        model: Trained CNN model.
        test_paths (numpy.array): Test image paths.
        test_labels (numpy.array): Test labels.
        result_dir (str): Directory to save results.
        class_names (list, optional): List of class names.

    Returns:
        dict: Evaluation metrics.
    """
    # Preprocessing function
    preprocess_fn = None
    if USE_GRAPHIC_PREPROCESSING:
        preprocess_fn = lambda img: apply_graphic_preprocessing(
            img,
            use_hair_removal=True,
            use_contrast_enhancement=True,
            use_segmentation=False,
            visualize=False
        )

    # Create test generator
    test_gen = MemoryEfficientDataGenerator(
        paths=test_paths,
        labels=test_labels,
        batch_size=BATCH_SIZE,
        model_name=CNN_MODEL,
        preprocess_fn=preprocess_fn,
        augment_fn=None,
        shuffle=False
    )

    # Calculate steps
    test_steps = math.ceil(len(test_paths) / BATCH_SIZE)

    # Collect predictions
    y_true = []
    y_pred = []
    y_pred_prob = []

    for i in range(test_steps):
        try:
            X_batch, y_batch = next(test_gen)

            # Get predictions
            pred_batch = model.predict(X_batch, verbose=0)

            # Convert one-hot encoded labels to class indices
            true_batch = np.argmax(y_batch, axis=1)
            pred_batch_cls = np.argmax(pred_batch, axis=1)

            # Append to lists
            y_true.extend(true_batch)
            y_pred.extend(pred_batch_cls)
            y_pred_prob.extend(pred_batch)
        except StopIteration:
            break

    # Convert lists to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)

    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True)

    # Print classification report
    print("\nTest Set Classification Report:")
    print(classification_report(y_true, y_pred))

    # Plot confusion matrix
    cm_plot_path = os.path.join(result_dir, "plots", "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_plot_path)

    # Save evaluation results
    results = {
        "accuracy": report["accuracy"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "class_report": report
    }

    # Save results to a text file
    with open(os.path.join(result_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Model: {CNN_MODEL}\n")
        f.write(f"Use Fine-tuning: {USE_FINE_TUNING}\n")
        f.write(f"Use Preprocessing: {USE_GRAPHIC_PREPROCESSING}\n")
        f.write(f"Use Data Augmentation: {USE_DATA_AUGMENTATION}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    return results


def run_kfold_cross_validation(all_paths, all_labels, result_dir, class_names=None):
    """
    Run K-fold cross-validation with a CNN classifier.

    Args:
        all_paths (numpy.array): All image paths.
        all_labels (numpy.array): All labels.
        result_dir (str): Directory to save results.
        class_names (list, optional): List of class names.

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
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_paths)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{NUM_KFOLDS}")
        print(f"{'=' * 50}")

        # Split data
        train_paths, val_paths = all_paths[train_idx], all_paths[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

        # Create model save path
        model_name = create_model_name(
            base_model_name=CNN_MODEL,
            mode='classifier',
            use_fine_tuning=USE_FINE_TUNING,
            use_preprocessing=USE_GRAPHIC_PREPROCESSING
        )
        model_save_path = os.path.join(
            result_dir,
            "models",
            f"{model_name}_fold_{fold + 1}.h5"
        )

        # Create fold result directory
        fold_dir = os.path.join(result_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "plots"), exist_ok=True)

        try:
            # Train model
            model, _ = run_single_fold_training(
                train_paths=train_paths,
                train_labels=train_labels,
                val_paths=val_paths,
                val_labels=val_labels,
                model_save_path=model_save_path,
                result_dir=fold_dir
            )

            # Evaluation
            y_true = []
            y_pred = []

            # Preprocessing function
            preprocess_fn = None
            if USE_GRAPHIC_PREPROCESSING:
                preprocess_fn = lambda img: apply_graphic_preprocessing(
                    img,
                    use_hair_removal=True,
                    use_contrast_enhancement=True,
                    use_segmentation=False,
                    visualize=False
                )

            # Create validation generator
            val_gen = MemoryEfficientDataGenerator(
                paths=val_paths,
                labels=val_labels,
                batch_size=BATCH_SIZE,
                model_name=CNN_MODEL,
                preprocess_fn=preprocess_fn,
                augment_fn=None,
                shuffle=False
            )

            # Calculate steps
            val_steps = math.ceil(len(val_paths) / BATCH_SIZE)

            # Collect predictions
            for i in range(val_steps):
                try:
                    X_batch, y_batch = next(val_gen)

                    # Get predictions
                    pred_batch = model.predict(X_batch, verbose=0)

                    # Convert one-hot encoded labels to class indices
                    true_batch = np.argmax(y_batch, axis=1)
                    pred_batch_cls = np.argmax(pred_batch, axis=1)

                    # Append to lists
                    y_true.extend(true_batch)
                    y_pred.extend(pred_batch_cls)
                except StopIteration:
                    break

            # Convert lists to arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # Add to overall predictions
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            # Calculate metrics
            report = classification_report(y_true, y_pred, output_dict=True)

            # Print classification report
            print(f"\nFold {fold + 1} Validation Set Classification Report:")
            print(classification_report(y_true, y_pred))

            # Plot confusion matrix
            cm_plot_path = os.path.join(fold_dir, "plots", "confusion_matrix.png")
            plot_confusion_matrix(y_true, y_pred, class_names, cm_plot_path)

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

            # Clean up
            clear_session()

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
    cm_plot_path = os.path.join(result_dir, "plots", "overall_confusion_matrix.png")
    plot_confusion_matrix(all_y_true, all_y_pred, class_names, cm_plot_path)

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
    with open(os.path.join(result_dir, "kfold_results.txt"), "w") as f:
        f.write(f"Model: {CNN_MODEL}\n")
        f.write(f"Use Fine-tuning: {USE_FINE_TUNING}\n")
        f.write(f"Use Preprocessing: {USE_GRAPHIC_PREPROCESSING}\n")
        f.write(f"Use Data Augmentation: {USE_DATA_AUGMENTATION}\n")
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


def train_and_evaluate(train_paths, train_labels, val_paths, val_labels,
                       test_paths, test_labels, class_names=None):
    """
    Train and evaluate a CNN classifier.

    Args:
        train_paths (numpy.array): Training image paths.
        train_labels (numpy.array): Training labels.
        val_paths (numpy.array): Validation image paths.
        val_labels (numpy.array): Validation labels.
        test_paths (numpy.array): Test image paths.
        test_labels (numpy.array): Test labels.
        class_names (list, optional): List of class names.

    Returns:
        tuple: (model, evaluation_results)
    """
    # Set up GPU memory
    setup_gpu_memory()

    # Create result directories
    result_dir = create_result_directories()
    print(f"Results will be saved to: {result_dir}")

    # Create model name and save path
    model_name = create_model_name(
        base_model_name=CNN_MODEL,
        mode='classifier',
        use_fine_tuning=USE_FINE_TUNING,
        use_preprocessing=USE_GRAPHIC_PREPROCESSING
    )
    model_save_path = os.path.join(result_dir, "models", f"{model_name}.h5")

    # Train model
    model, _ = run_single_fold_training(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        model_save_path=model_save_path,
        result_dir=result_dir
    )

    # Evaluate on test set
    test_results = evaluate_model(
        model=model,
        test_paths=test_paths,
        test_labels=test_labels,
        result_dir=result_dir,
        class_names=class_names
    )

    return model, test_results


def run_cross_validation(all_data_paths, all_data_labels, class_names=None):
    """
    Run K-fold cross-validation on all data.

    Args:
        all_data_paths (numpy.array): All image paths.
        all_data_labels (numpy.array): All labels.
        class_names (list, optional): List of class names.

    Returns:
        list: List of evaluation results for each fold.
    """
    # Set up GPU memory
    setup_gpu_memory()

    # Create result directories
    result_dir = create_result_directories()
    print(f"K-fold cross-validation results will be saved to: {result_dir}")

    # Run K-fold cross-validation
    fold_results = run_kfold_cross_validation(
        all_paths=all_data_paths,
        all_labels=all_data_labels,
        result_dir=result_dir,
        class_names=class_names
    )

    return fold_results


def run_cnn_classifier_pipeline(train_files_path, val_files_path, test_files_path,
                                run_kfold=False, class_names=None):
    """
    Run the CNN classifier pipeline.

    Args:
        train_files_path (str): Path to training files list.
        val_files_path (str): Path to validation files list.
        test_files_path (str): Path to test files list.
        run_kfold (bool): Whether to run K-fold cross-validation.
        class_names (list, optional): List of class names.

    Returns:
        dict: Results of the pipeline.
    """
    # Load data paths and labels
    train_paths, train_labels = load_paths_labels(train_files_path)
    val_paths, val_labels = load_paths_labels(val_files_path)
    test_paths, test_labels = load_paths_labels(test_files_path)

    results = {}

    if run_kfold:
        # Combine training and validation sets for cross-validation
        all_data_paths = np.concatenate([train_paths, val_paths])
        all_data_labels = np.concatenate([train_labels, val_labels])

        # Run cross-validation
        fold_results = run_cross_validation(
            all_data_paths=all_data_paths,
            all_data_labels=all_data_labels,
            class_names=class_names
        )

        results['k_fold'] = fold_results
    else:
        # Train and evaluate on fixed splits
        model, test_results = train_and_evaluate(
            train_paths=train_paths,
            train_labels=train_labels,
            val_paths=val_paths,
            val_labels=val_labels,
            test_paths=test_paths,
            test_labels=test_labels,
            class_names=class_names
        )

        results['test_evaluation'] = test_results

    return results