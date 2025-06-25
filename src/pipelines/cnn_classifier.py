"""
End-to-end CNN classification pipeline.
Handles training, evaluation, and K-fold cross-validation.
"""

import datetime
import gc
import math
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.backend import clear_session

sys.path.append('..')
from config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    USE_GRAPHIC_PREPROCESSING,
    USE_DATA_AUGMENTATION,
    USE_FEATURE_PREPROCESSING,
    USE_FINE_TUNING,
    NUM_KFOLDS,
    CNN_MODEL,
    RESULTS_DIR,
    USE_HAIR_REMOVAL,
    USE_ENHANCED_CONTRAST,
    FINE_TUNING_AT_LAYER,
    NUM_ITERATIONS,
    NUM_CLASSES,
    NUM_FINAL_MODELS
)

from utils.data_loaders import load_paths_labels, MemoryEfficientDataGenerator
from preprocessing.data.augmentation import AugmentationFactory
from models.cnn_models import load_or_create_cnn, get_callbacks, create_model_name
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


def create_result_directories(base_dir=RESULTS_DIR):
    """Create directories for saving results."""
    str_hair       = "hair_removal_" if USE_HAIR_REMOVAL else ""
    str_contrast   = "contrast_" if USE_ENHANCED_CONTRAST else ""
    str_graphic    = f"{str_contrast}{str_hair}" if USE_GRAPHIC_PREPROCESSING else ""
    str_augment    = "use_augmentation_" if USE_DATA_AUGMENTATION else ""
    str_preprocess = "use_feature_preprocess_" if USE_FEATURE_PREPROCESSING else ""
    result_dir     = os.path.join(base_dir, f"cnn_classifier_{CNN_MODEL}_{str_graphic}{str_augment}{str_preprocess}")

    # Create subdirectories
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "models"), exist_ok=True)

    return result_dir



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

    train_gen = MemoryEfficientDataGenerator(
        paths=train_paths,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        model_name=CNN_MODEL,
        augment_fn=augment_fn,
        shuffle=True
    )

    val_gen = MemoryEfficientDataGenerator(
        paths=val_paths,
        labels=val_labels,
        batch_size=BATCH_SIZE,
        model_name=CNN_MODEL,
        augment_fn=None,
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

    # Create test generator
    test_gen = MemoryEfficientDataGenerator(
        paths=test_paths,
        labels=test_labels,
        batch_size=BATCH_SIZE,
        model_name=CNN_MODEL,
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
        'hyperparameters': None  # Store hyperparameters
    }

    # Run multiple iterations
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'=' * 50}")
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'=' * 50}")

        # Create iteration directory
        iter_dir = os.path.join(result_dir, f"iteration_{iteration + 1}")
        os.makedirs(iter_dir, exist_ok=True)
        os.makedirs(os.path.join(iter_dir, "plots"), exist_ok=True)

        # Initialize StratifiedKFold with a different random state for each iteration
        skf = StratifiedKFold(n_splits=NUM_KFOLDS, shuffle=True, random_state=42 + iteration)

        # List to store evaluation results for this iteration
        fold_results = []

        # Dictionary to collect predictions for this iteration
        iteration_y_true = []
        iteration_y_pred = []

        # Run each fold
        # Note: We need to use integer labels with StratifiedKFold, not one-hot encoded
        # Get integer labels if they're one-hot encoded
        if len(all_labels.shape) > 1 and all_labels.shape[1] > 1:
            stratify_labels = np.argmax(all_labels, axis=1)
        else:
            stratify_labels = all_labels

        # Run each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, stratify_labels)):
            print(f"\n{'=' * 40}")
            print(f"Iteration {iteration + 1}, Fold {fold + 1}/{NUM_KFOLDS}")
            print(f"{'=' * 40}")

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
                iter_dir,
                "models",
                f"{model_name}_iter_{iteration + 1}_fold_{fold + 1}.h5"
            )

            # Create fold result directory
            fold_dir = os.path.join(iter_dir, f"fold_{fold + 1}")
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

                # Create validation generator
                val_gen = MemoryEfficientDataGenerator(
                    paths=val_paths,
                    labels=val_labels,
                    batch_size=BATCH_SIZE,
                    model_name=CNN_MODEL,
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

                # Add to iteration predictions
                iteration_y_true.extend(y_true)
                iteration_y_pred.extend(y_pred)

                # Calculate metrics
                report = classification_report(y_true, y_pred, output_dict=True)

                # Print classification report
                print(f"\nIteration {iteration + 1}, Fold {fold + 1} Validation Set Classification Report:")
                print(classification_report(y_true, y_pred))

                if report['accuracy'] > best_model_metrics['accuracy'] or \
                        (report['accuracy'] == best_model_metrics['accuracy'] and
                         report['macro avg']['f1-score'] > best_model_metrics['macro_avg_f1']):
                    # Store model hyperparameters for CNN models
                    hyperparameters = {
                        'model_name': CNN_MODEL,
                        'batch_size': BATCH_SIZE,
                        'fine_tuning': USE_FINE_TUNING,
                        'fine_tuning_at_layer': FINE_TUNING_AT_LAYER.get(CNN_MODEL),
                        'use_augmentation': USE_DATA_AUGMENTATION,
                        'use_graphic_preprocessing': USE_GRAPHIC_PREPROCESSING,
                        'use_hair_removal': USE_HAIR_REMOVAL,
                        'use_enhanced_contrast': USE_ENHANCED_CONTRAST
                    }

                    best_model_metrics = {
                        'iteration': iteration + 1,
                        'fold': fold + 1,
                        'accuracy': report['accuracy'],
                        'macro_avg_f1': report['macro avg']['f1-score'],
                        'model_path': model_save_path,
                        'hyperparameters': hyperparameters
                    }

                # Store fold results
                fold_result = {
                    'iteration': iteration + 1,
                    'fold': fold + 1,
                    'accuracy': report['accuracy'],
                    'macro_avg_precision': report['macro avg']['precision'],
                    'macro_avg_recall': report['macro avg']['recall'],
                    'macro_avg_f1': report['macro avg']['f1-score'],
                    'class_report': report
                }

                fold_results.append(fold_result)

                # Clean up
                clear_session()

            except Exception as e:
                print(f"Error in iteration {iteration + 1}, fold {fold + 1}: {e}")
                continue

        # Calculate overall metrics for this iteration
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
        avg_accuracy = np.mean([res['accuracy'] for res in fold_results])
        avg_precision = np.mean([res['macro_avg_precision'] for res in fold_results])
        avg_recall = np.mean([res['macro_avg_recall'] for res in fold_results])
        avg_f1 = np.mean([res['macro_avg_f1'] for res in fold_results])

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
    with open(os.path.join(result_dir, "overall_results.txt"), "w") as f:
        f.write(f"Model: {CNN_MODEL}\n")
        f.write(f"Use Fine-tuning: {USE_FINE_TUNING}\n")
        f.write(f"Use Preprocessing: {USE_GRAPHIC_PREPROCESSING}\n")
        f.write(f"Use Data Augmentation: {USE_DATA_AUGMENTATION}\n")
        f.write(f"Number of Folds: {NUM_KFOLDS}\n")
        f.write(f"Number of Iterations: {NUM_ITERATIONS}\n\n")

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
        'best_hyperparameters': best_model_metrics['hyperparameters'],
        'result_dir': result_dir
    }


def train_final_model_cnn(all_data_paths, all_data_labels, best_hyperparameters, result_dir, class_names=None):
    """
    Train a final CNN model on all training data using the best hyperparameters.

    Args:
        all_data_paths: Combined training and validation paths
        all_data_labels: Combined training and validation labels
        best_hyperparameters: Best hyperparameters found during cross-validation
        result_dir: Directory to save results
        class_names: List of class names

    Returns:
        Final trained model and evaluation results
    """
    print("\n" + "=" * 60)
    print("Training Final CNN Model on All Training Data")
    print("=" * 60)

    # Create final model directory
    final_model_dir = os.path.join(result_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    os.makedirs(os.path.join(final_model_dir, "plots"), exist_ok=True)

    # Final model save path
    final_model_path = os.path.join(final_model_dir, "final_cnn_model.h5")

    # Create or load the model with best hyperparameters
    model_name = best_hyperparameters['model_name']
    use_fine_tuning = best_hyperparameters['fine_tuning']

    # Create model using best hyperparameters
    model, _ = load_or_create_cnn(
        model_name=model_name,
        mode='classifier',
        fine_tune=use_fine_tuning,
        save_path=None  # Don't try to load existing
    )

    # Define augmentation function if used in best model
    augment_fn = None
    if best_hyperparameters['use_augmentation']:
        augmentation = AugmentationFactory.get_medium_augmentation()
        augment_fn = lambda img: augmentation(image=img)['image']

    # Create data generators
    batch_size = best_hyperparameters['batch_size']

    train_gen = MemoryEfficientDataGenerator(
        paths=all_data_paths,
        labels=all_data_labels,
        batch_size=batch_size,
        model_name=model_name,
        augment_fn=augment_fn,
        shuffle=True
    )

    # For validation during training, use a small subset of the training data
    # This is just to monitor training progress, not for model selection
    val_size = min(1000, len(all_data_paths) // 5)  # 20% or max 1000 samples
    val_indices = np.random.choice(len(all_data_paths), val_size, replace=False)
    val_paths = all_data_paths[val_indices]
    val_labels = all_data_labels[val_indices]

    val_gen = MemoryEfficientDataGenerator(
        paths=val_paths,
        labels=val_labels,
        batch_size=batch_size,
        model_name=model_name,
        augment_fn=None,
        shuffle=False
    )

    # Calculate steps per epoch
    steps_per_epoch = math.ceil(len(all_data_paths) / batch_size)
    validation_steps = math.ceil(len(val_paths) / batch_size)

    # TensorBoard log directory
    log_dir = os.path.join(final_model_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Get callbacks
    callbacks = get_callbacks(final_model_path, log_dir)

    # Train the model on all data
    print(f"Training final CNN model on all {len(all_data_paths)} samples...")
    history = model.fit(
        train_gen.get_keras_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=NUM_EPOCHS,
        validation_data=val_gen.get_keras_generator(),
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Final CNN model trained and saved to: {final_model_path}")

    return model, final_model_dir


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


def train_multiple_final_cnn_models(all_data_paths, all_data_labels, best_hyperparameters,
                                    result_dir, class_names=None, num_models=10):
    """
    Train multiple final CNN models on all training data using the best hyperparameters.

    Args:
        all_data_paths: Combined training and validation paths
        all_data_labels: Combined training and validation labels
        best_hyperparameters: Best hyperparameters found during cross-validation
        result_dir: Directory to save results
        class_names: List of class names
        num_models: Number of models to train

    Returns:
        List of trained models and their directories
    """
    print("\n" + "=" * 60)
    print(f"Training {num_models} Final CNN Models on All Training Data")
    print("=" * 60)

    # Create final models directory
    final_models_dir = os.path.join(result_dir, "final_models")
    os.makedirs(final_models_dir, exist_ok=True)

    # Get augmentation pipeline if enabled
    augment_fn = None
    if best_hyperparameters.get('use_augmentation', USE_DATA_AUGMENTATION):
        augmentation = AugmentationFactory.get_medium_augmentation()
        augment_fn = lambda img: augmentation(image=img)['image']

    trained_models = []

    for model_idx in range(num_models):
        print(f"\n{'=' * 50}")
        print(f"Training CNN Model {model_idx + 1}/{num_models}")
        print(f"{'=' * 50}")

        # Create model-specific directory
        model_dir = os.path.join(final_models_dir, f"model_{model_idx + 1}")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True)

        # Model save path
        model_path = os.path.join(model_dir, "final_cnn_model.h5")

        # Create data generators with different random shuffling for each model
        train_gen = MemoryEfficientDataGenerator(
            paths=all_data_paths,
            labels=all_data_labels,
            batch_size=best_hyperparameters.get('batch_size', BATCH_SIZE),
            model_name=best_hyperparameters['model_name'],
            augment_fn=augment_fn,
            shuffle=True
        )

        # For validation during training, use a small subset
        val_size = min(1000, len(all_data_paths) // 5)
        val_indices = np.random.RandomState(42 + model_idx).choice(
            len(all_data_paths), val_size, replace=False
        )
        val_paths = all_data_paths[val_indices]
        val_labels = all_data_labels[val_indices]

        val_gen = MemoryEfficientDataGenerator(
            paths=val_paths,
            labels=val_labels,
            batch_size=best_hyperparameters.get('batch_size', BATCH_SIZE),
            model_name=best_hyperparameters['model_name'],
            augment_fn=None,
            shuffle=False
        )

        # Calculate steps
        steps_per_epoch = math.ceil(len(all_data_paths) / best_hyperparameters.get('batch_size', BATCH_SIZE))
        validation_steps = math.ceil(len(val_paths) / best_hyperparameters.get('batch_size', BATCH_SIZE))

        # TensorBoard log directory
        log_dir = os.path.join(model_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Create model
        model, _ = load_or_create_cnn(
            model_name=best_hyperparameters['model_name'],
            mode='classifier',
            fine_tune=best_hyperparameters.get('fine_tuning', USE_FINE_TUNING),
            save_path=None  # Don't load existing
        )

        # Get callbacks
        callbacks = get_callbacks(model_path, log_dir)

        # Train the model
        print(f"Training model {model_idx + 1} on all {len(all_data_paths)} samples...")
        history = model.fit(
            train_gen.get_keras_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=NUM_EPOCHS,
            validation_data=val_gen.get_keras_generator(),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        trained_models.append({
            'model': model,
            'model_path': model_path,
            'model_dir': model_dir,
            'model_idx': model_idx + 1,
            'history': history
        })

        # Clear memory
        clear_session()
        gc.collect()

    print(f"\nAll {num_models} CNN models trained successfully!")

    return trained_models, final_models_dir


def evaluate_multiple_final_cnn_models(trained_models, test_paths, test_labels,
                                       result_dir, class_names=None):
    """
    Evaluate multiple final CNN models on the test set and perform statistical analysis.

    Args:
        trained_models: List of trained model dictionaries
        test_paths: Test image paths
        test_labels: Test labels
        result_dir: Directory to save results
        class_names: List of class names

    Returns:
        Dictionary with evaluation results and statistical analysis
    """
    from scipy import stats
    import pandas as pd

    print("\n" + "=" * 60)
    print("Evaluating Multiple Final CNN Models on Test Set")
    print("=" * 60)

    # Results storage
    all_results = {
        'model_metrics': [],
        'predictions': [],
        'accuracies': [],
        'f1_scores': [],
        'precisions': [],
        'recalls': [],
        'class_metrics': []
    }

    # Evaluate each model
    for model_info in trained_models:
        model = model_info['model']
        model_idx = model_info['model_idx']
        model_dir = model_info['model_dir']

        print(f"\nEvaluating CNN Model {model_idx}/{len(trained_models)}...")

        # Create test generator
        test_gen = MemoryEfficientDataGenerator(
            paths=test_paths,
            labels=test_labels,
            batch_size=BATCH_SIZE,
            model_name=CNN_MODEL,
            augment_fn=None,
            shuffle=False
        )

        # Calculate steps
        test_steps = math.ceil(len(test_paths) / BATCH_SIZE)

        # Collect predictions
        y_true = []
        y_pred = []

        for i in range(test_steps):
            try:
                X_batch, y_batch = next(test_gen)
                pred_batch = model.predict(X_batch, verbose=0)

                true_batch = np.argmax(y_batch, axis=1)
                pred_batch_cls = np.argmax(pred_batch, axis=1)

                y_true.extend(true_batch)
                y_pred.extend(pred_batch_cls)
            except StopIteration:
                break

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate metrics
        test_report = classification_report(y_true, y_pred, output_dict=True)

        # Store predictions and metrics
        all_results['predictions'].append(y_pred)
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
        for class_idx in range(len(class_names) if class_names else NUM_CLASSES):
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
            f.write(f"Model: {CNN_MODEL}\n")
            f.write(f"Use Fine-tuning: {USE_FINE_TUNING}\n")
            f.write(f"Use Preprocessing: {USE_GRAPHIC_PREPROCESSING}\n")
            f.write(f"Use Data Augmentation: {USE_DATA_AUGMENTATION}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred))
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_true, y_pred)))

    # Statistical Analysis
    print("\n" + "=" * 50)
    print("Statistical Analysis of CNN Model Performance")
    print("=" * 50)

    # Convert to numpy arrays
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
        f.write("Statistical Analysis of Multiple Final CNN Models\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of models: {len(trained_models)}\n\n")

        for metric_name, metric_stats in stats_results.items():
            f.write(f"{metric_name.upper()}:\n")
            f.write(f"  Mean ± Std: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}\n")
            f.write(f"  Median: {metric_stats['median']:.4f}\n")
            f.write(f"  Min/Max: {metric_stats['min']:.4f} / {metric_stats['max']:.4f}\n")
            f.write(f"  95% CI: [{metric_stats['95_ci'][0]:.4f}, {metric_stats['95_ci'][1]:.4f}]\n\n")

    # Create DataFrames
    df_results = pd.DataFrame(all_results['model_metrics'])
    df_summary = df_results[['model_idx', 'accuracy', 'macro_avg_precision',
                             'macro_avg_recall', 'macro_avg_f1']]

    # Save as CSV
    csv_path = os.path.join(result_dir, "model_performance_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    # Save per-class metrics
    df_class_metrics = None
    class_stats = {}

    if all_results['class_metrics']:
        df_class_metrics = pd.DataFrame(all_results['class_metrics'])
        class_csv_path = os.path.join(result_dir, "per_class_metrics.csv")
        df_class_metrics.to_csv(class_csv_path, index=False)

        # Calculate per-class statistics
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

        # Save per-class statistics
        class_stats_path = os.path.join(result_dir, "per_class_statistics.txt")
        with open(class_stats_path, "w") as f:
            f.write("Per-Class Performance Statistics\n")
            f.write("=" * 50 + "\n\n")

            for class_name, stats in class_stats.items():
                f.write(f"{class_name}:\n")
                f.write(f"  F1-Score: {stats['f1_score']['mean']:.4f} ± {stats['f1_score']['std']:.4f}\n")
                f.write(f"  Precision: {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}\n")
                f.write(f"  Recall: {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}\n")
                f.write(f"  Range: [{stats['f1_score']['min']:.4f}, {stats['f1_score']['max']:.4f}]\n\n")

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
        'class_metrics_df': df_class_metrics,
        'class_statistics': class_stats
    }

def run_cnn_classifier_pipeline(train_files_path, val_files_path, test_files_path,
                                run_kfold=False, class_names=None, skip_training=False):
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

    if skip_training:
        print("\nStarting training of multiple final CNN models...")
        result_dir = create_result_directories()
        all_data_paths = np.concatenate([train_paths, val_paths])
        all_data_labels = np.concatenate([train_labels, val_labels])
        hyperparameters = {
            'model_name': CNN_MODEL,
            'batch_size': BATCH_SIZE,
            'fine_tuning': USE_FINE_TUNING,
            'fine_tuning_at_layer': FINE_TUNING_AT_LAYER.get(CNN_MODEL),
            'use_augmentation': USE_DATA_AUGMENTATION,
            'use_graphic_preprocessing': USE_GRAPHIC_PREPROCESSING,
            'use_hair_removal': USE_HAIR_REMOVAL,
            'use_enhanced_contrast': USE_ENHANCED_CONTRAST
        }
        trained_models, final_models_dir = train_multiple_final_cnn_models(
            all_data_paths=all_data_paths,
            all_data_labels=all_data_labels,
            best_hyperparameters=hyperparameters,
            result_dir=result_dir,
            class_names=class_names,
            num_models=NUM_FINAL_MODELS
        )
        print(f"All {NUM_FINAL_MODELS} final CNN models trained and saved in: {final_models_dir}")

        # Evaluate all final models on test set
        print("\nEvaluating multiple final CNN models on test set...")

        eval_results = evaluate_multiple_final_cnn_models(
            trained_models=trained_models,
            test_paths=test_paths,
            test_labels=test_labels,
            result_dir=final_models_dir,
            class_names=class_names
        )
        return
    if run_kfold:
        # Combine training and validation sets for cross-validation
        all_data_paths = np.concatenate([train_paths, val_paths])
        all_data_labels = np.concatenate([train_labels, val_labels])

        # Run cross-validation
        cv_results = run_cross_validation(
            all_data_paths=all_data_paths,
            all_data_labels=all_data_labels,
            class_names=class_names
        )

        results['k_fold'] = cv_results['fold_results']
        results['best_model_info'] = cv_results['best_model_info']

        # Save detailed fold results
        save_fold_results(
            fold_results=cv_results['fold_results'],
            result_dir=cv_results['result_dir'],
            classifier_name='CNN'
        )

        # Train multiple final models with all training data using best hyperparameters
        try:
            print("\nStarting training of multiple final CNN models...")

            trained_models, final_models_dir = train_multiple_final_cnn_models(
                all_data_paths=all_data_paths,
                all_data_labels=all_data_labels,
                best_hyperparameters=cv_results['best_hyperparameters'],
                result_dir=cv_results['result_dir'],
                class_names=class_names,
                num_models=NUM_FINAL_MODELS
            )
            print(f"All {NUM_FINAL_MODELS} final CNN models trained and saved in: {final_models_dir}")

            # Evaluate all final models on test set
            print("\nEvaluating multiple final CNN models on test set...")

            eval_results = evaluate_multiple_final_cnn_models(
                trained_models=trained_models,
                test_paths=test_paths,
                test_labels=test_labels,
                result_dir=final_models_dir,
                class_names=class_names
            )

            # Store comprehensive results
            results['final_models'] = trained_models
            results['final_models_evaluation'] = eval_results
            results['statistical_analysis'] = eval_results['statistics']
            results['class_statistics'] = eval_results.get('class_statistics', None)

            # Save comprehensive summary
            summary_path = os.path.join(cv_results['result_dir'], "complete_experiment_summary.txt")
            with open(summary_path, "w") as f:
                f.write("Complete CNN Experiment Summary\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"CNN Model: {CNN_MODEL}\n")
                f.write(f"Use Fine-tuning: {USE_FINE_TUNING}\n")
                f.write(f"Use Data Augmentation: {USE_DATA_AUGMENTATION}\n")
                f.write(f"Use Preprocessing: {USE_GRAPHIC_PREPROCESSING}\n")
                if USE_GRAPHIC_PREPROCESSING:
                    f.write(f"  Hair Removal: {USE_HAIR_REMOVAL}\n")
                    f.write(f"  Contrast Enhancement: {USE_ENHANCED_CONTRAST}\n")
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

                if 'class_statistics' in results and results['class_statistics']:
                    f.write(f"\nPer-Class Performance Summary (Mean ± Std):\n")
                    for class_name, class_stats in results['class_statistics'].items():
                        f.write(f"\n{class_name}:\n")
                        f.write(
                            f"  F1-Score: {class_stats['f1_score']['mean']:.4f} ± {class_stats['f1_score']['std']:.4f}\n")
                        f.write(
                            f"  Precision: {class_stats['precision']['mean']:.4f} ± {class_stats['precision']['std']:.4f}\n")
                        f.write(f"  Recall: {class_stats['recall']['mean']:.4f} ± {class_stats['recall']['std']:.4f}\n")

        except Exception as e:
            import traceback
            print(f"ERROR in training multiple final CNN models: {e}")
            traceback.print_exc()

            # Fallback to single final model
            print("\nFalling back to single final model training...")
            final_model, final_model_dir = train_final_model_cnn(
                all_data_paths=all_data_paths,
                all_data_labels=all_data_labels,
                best_hyperparameters=cv_results['best_hyperparameters'],
                result_dir=cv_results['result_dir'],
                class_names=class_names
            )

            # Evaluate final model on test set
            print("\nEvaluating final model on test set...")
            test_results = evaluate_model(
                model=final_model,
                test_paths=test_paths,
                test_labels=test_labels,
                result_dir=final_model_dir,
                class_names=class_names
            )

            results['final_model_test_results'] = test_results