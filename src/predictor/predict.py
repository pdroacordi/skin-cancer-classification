"""
Standalone script to collect predictions from all trained models.
Can be run from any directory.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report
import joblib
import gc
import cv2
import argparse
from tensorflow.keras.utils import to_categorical

# Constants
BATCH_SIZE = 8
IMG_SIZE = (299, 299, 3)
NUM_CLASSES = 7


def setup_gpu_memory():
    """Set up GPU memory growth to avoid OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")


def load_paths_labels(file_path):
    """Load image paths and labels from a text file."""
    paths, labels = [], []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                path, label = line.strip().split('\t')
                if os.path.exists(path):
                    paths.append(path)
                    labels.append(int(label))
                else:
                    print(f"Warning: Image at path {path} does not exist.")
            except ValueError:
                print(f"Line {idx + 1} is malformed: {line}")
    return np.array(paths), np.array(labels)


def apply_model_preprocessing(image, model_name):
    """Apply model-specific preprocessing to an image."""
    # Import preprocessing functions
    from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
    from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
    from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception

    if model_name == "VGG19":
        return preprocess_vgg19(image.copy())
    elif model_name == "Inception":
        return preprocess_inception(image.copy())
    elif model_name == "ResNet":
        return preprocess_resnet(image.copy())
    elif model_name == "Xception":
        return preprocess_xception(image.copy())
    else:
        return image.astype(np.float32) / 255.0


def extract_cnn_name(directory_name):
    """Extract CNN name from directory name."""
    cnn_models = ['VGG19', 'Inception', 'ResNet', 'Xception']
    for model in cnn_models:
        if model.lower() in directory_name.lower():
            return model
    return 'Unknown'


def extract_fold_from_filename(filename):
    """Extract fold number from filename."""
    import re
    match = re.search(r'fold_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 1


def predict_cnn_on_test(model, test_paths, test_labels, cnn_name, batch_size=8):
    """Get predictions from a CNN model on test data."""
    predictions = []
    probabilities = []

    # Process in batches
    for i in range(0, len(test_paths), batch_size):
        batch_paths = test_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            # Load and preprocess image
            img = cv2.imread(path)
            if img is None:
                print(f"Error loading image: {path}")
                continue

            # Resize
            img = cv2.resize(img, IMG_SIZE[:2])

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply model-specific preprocessing
            img = apply_model_preprocessing(img, cnn_name)

            batch_images.append(img)

        if batch_images:
            batch_images = np.array(batch_images)

            # Get predictions
            batch_proba = model.predict(batch_images, verbose=0)
            batch_pred = np.argmax(batch_proba, axis=1)

            predictions.extend(batch_pred)
            probabilities.extend(batch_proba)

        # Clear memory
        del batch_images
        gc.collect()

    return test_labels[:len(predictions)], np.array(predictions), np.array(probabilities)


def predict_ml_on_features(model, test_features, test_labels):
    """Get predictions from a ML model on test features."""
    # Get predictions
    y_pred = model.predict(test_features)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(test_features)
    else:
        # Create dummy probabilities
        y_pred_proba = np.zeros((len(y_pred), NUM_CLASSES))
        y_pred_proba[np.arange(len(y_pred)), y_pred] = 1.0

    return test_labels, y_pred, y_pred_proba


def extract_features_simple(cnn_model, test_paths, cnn_name, batch_size=8):
    """Extract features using a CNN model."""
    # Create feature extractor from CNN
    # Find the global average pooling or last conv layer
    feature_layer = None
    for layer in reversed(cnn_model.layers):
        if 'global' in layer.name or 'pool' in layer.name:
            feature_layer = layer
            break
        elif 'conv' in layer.name:
            feature_layer = layer
            break

    if feature_layer is None:
        # Use the layer before the last dense layer
        for i in range(len(cnn_model.layers)-1, -1, -1):
            if 'dense' not in cnn_model.layers[i].name.lower():
                feature_layer = cnn_model.layers[i]
                break

    # Create feature extractor
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=feature_layer.output
    )

    features = []

    # Extract features in batches
    for i in range(0, len(test_paths), batch_size):
        batch_paths = test_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            # Load and preprocess image
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE[:2])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = apply_model_preprocessing(img, cnn_name)
            batch_images.append(img)

        if batch_images:
            batch_images = np.array(batch_images)
            batch_features = feature_extractor.predict(batch_images, verbose=0)

            # Handle different output shapes
            if len(batch_features.shape) > 2:
                # Apply global average pooling
                batch_features = np.mean(batch_features, axis=(1, 2))

            features.extend(batch_features)

        del batch_images
        gc.collect()

    return np.array(features)


def collect_all_predictions(test_files_path, results_dir, output_dir):
    """Main function to collect all predictions."""
    setup_gpu_memory()

    # Load test data
    test_paths, test_labels = load_paths_labels(test_files_path)
    print(f"Loaded {len(test_paths)} test samples")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Results storage
    all_results = []
    summary_data = []

    # Feature cache for ML models
    feature_cache = {}

    # 1. Process CNN End-to-End Models
    print("\n" + "="*60)
    print("Processing CNN End-to-End Models")
    print("="*60)

    cnn_pattern = os.path.join(results_dir, 'cnn_classifier_*', 'iteration_*', 'models', '*.h5')
    cnn_models = glob.glob(cnn_pattern)

    print(f"Found {len(cnn_models)} CNN models")

    for model_path in cnn_models:
        try:
            # Extract metadata from path
            parts = Path(model_path).parts
            cnn_dir = next(p for p in parts if p.startswith('cnn_classifier_'))
            cnn_name = extract_cnn_name(cnn_dir)
            iter_dir = next(p for p in parts if p.startswith('iteration_'))
            iteration = int(iter_dir.split('_')[1])
            fold = extract_fold_from_filename(os.path.basename(model_path))

            print(f"\nProcessing {cnn_name} - Iteration {iteration}, Fold {fold}")

            # Load model
            model = tf.keras.models.load_model(model_path)

            # Get predictions
            y_true, y_pred, y_proba = predict_cnn_on_test(
                model, test_paths, test_labels, cnn_name
            )

            # Calculate metrics
            report = classification_report(y_true, y_pred, output_dict=True)

            class_metrics = {
                label: {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'support': metrics['support']
                }
                for label, metrics in report.items()
                if label not in ['accuracy', 'macro avg', 'weighted avg']
            }

            # Store results
            result = {
                'model_type': 'cnn',
                'algorithm': cnn_name,
                'iteration': iteration,
                'fold': fold,
                'model_path': model_path,
                'accuracy': report['accuracy'],
                'macro_avg_precision': report['macro avg']['precision'],
                'macro_avg_recall': report['macro avg']['recall'],
                'macro_avg_f1': report['macro avg']['f1-score'],
                'y_true': y_true,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'per_class_metrics': class_metrics
            }



            all_results.append(result)

            summary_data.append({
                'model_type': 'cnn',
                'algorithm': cnn_name,
                'iteration': iteration,
                'fold': fold,
                'accuracy': report['accuracy'],
                'macro_avg_precision': report['macro avg']['precision'],
                'macro_avg_recall': report['macro avg']['recall'],
                'macro_avg_f1': report['macro avg']['f1-score'],
                'per_class_metrics': class_metrics
            })

            # Save predictions
            pred_file = f"cnn_{cnn_name}_iter{iteration}_fold{fold}.npz"
            np.savez(
                os.path.join(output_dir, pred_file),
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba
            )

            # Clean up
            del model
            tf.keras.backend.clear_session()
            gc.collect()

        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            continue

    # 2. Process Feature Extraction + ML Models
    print("\n" + "="*60)
    print("Processing Feature Extraction + ML Models")
    print("="*60)

    ml_pattern = os.path.join(results_dir, 'feature_extraction_*', '*', 'iteration_*', 'fold_*', 'models', '*.joblib')
    ml_models = glob.glob(ml_pattern)

    print(f"Found {len(ml_models)} ML models")

    # First, find a final CNN model for each extractor type to extract features
    cnn_final_models = {}
    for cnn_name in ['VGG19', 'Inception', 'ResNet', 'Xception']:
        # Look for final CNN model
        final_pattern = os.path.join(results_dir, f'cnn_classifier_{cnn_name}*', 'final_model', 'final_cnn_model.h5')
        final_models = glob.glob(final_pattern)

        if final_models:
            cnn_final_models[cnn_name] = final_models[0]
            print(f"Found final CNN model for {cnn_name}: {final_models[0]}")

    for model_path in ml_models:
        try:
            # Extract metadata from path
            parts = Path(model_path).parts
            fe_dir = next(p for p in parts if p.startswith('feature_extraction_'))
            extractor_name = extract_cnn_name(fe_dir)

            # Find classifier name
            classifier_name = None
            for part in parts:
                if part.lower() in ['randomforest', 'xgboost', 'adaboost', 'extratrees']:
                    classifier_name = part
                    break

            if not classifier_name:
                continue

            iter_dir = next(p for p in parts if p.startswith('iteration_'))
            iteration = int(iter_dir.split('_')[1])
            fold_dir = next(p for p in parts if p.startswith('fold_'))
            fold = int(fold_dir.split('_')[1])

            print(f"\nProcessing {extractor_name}+{classifier_name} - Iteration {iteration}, Fold {fold}")

            # Get or extract features
            if extractor_name not in feature_cache:
                # Look for cached features first
                feature_file = os.path.join(
                    results_dir,
                    f'feature_extraction_{extractor_name}*',
                    'features',
                    'test_features.npz'
                )
                feature_files = glob.glob(feature_file)

                if feature_files:
                    print(f"Loading cached features from {feature_files[0]}")
                    data = np.load(feature_files[0])
                    test_features = data['features']
                    if 'labels' in data:
                        test_labels_fe = data['labels']
                    else:
                        test_labels_fe = test_labels
                else:
                    # Extract features using CNN model
                    if extractor_name in cnn_final_models:
                        print(f"Extracting features using {extractor_name} CNN...")
                        cnn_model = tf.keras.models.load_model(cnn_final_models[extractor_name])
                        test_features = extract_features_simple(
                            cnn_model, test_paths, extractor_name
                        )
                        test_labels_fe = test_labels[:len(test_features)]

                        # Clean up
                        del cnn_model
                        tf.keras.backend.clear_session()
                    else:
                        print(f"No CNN model found for {extractor_name}, skipping...")
                        continue

                feature_cache[extractor_name] = (test_features, test_labels_fe)
            else:
                test_features, test_labels_fe = feature_cache[extractor_name]

            # Load ML model
            model = joblib.load(model_path)

            # Get predictions
            y_true, y_pred, y_proba = predict_ml_on_features(
                model, test_features, test_labels_fe
            )

            # Calculate metrics
            report = classification_report(y_true, y_pred, output_dict=True)

            class_metrics = {
                label: {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'support': metrics['support']
                }
                for label, metrics in report.items()
                if label not in ['accuracy', 'macro avg', 'weighted avg']
            }

            # Store results
            result = {
                'model_type': 'feature_extraction',
                'algorithm': f"{extractor_name}+{classifier_name}",
                'iteration': iteration,
                'fold': fold,
                'model_path': model_path,
                'accuracy': report['accuracy'],
                'macro_avg_precision': report['macro avg']['precision'],
                'macro_avg_recall': report['macro avg']['recall'],
                'macro_avg_f1': report['macro avg']['f1-score'],
                'y_true': y_true,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'per_class_metrics': class_metrics
            }

            all_results.append(result)

            summary_data.append({
                'model_type': 'feature_extraction',
                'algorithm': f"{extractor_name}+{classifier_name}",
                'iteration': iteration,
                'fold': fold,
                'accuracy': report['accuracy'],
                'macro_avg_precision': report['macro avg']['precision'],
                'macro_avg_recall': report['macro avg']['recall'],
                'macro_avg_f1': report['macro avg']['f1-score'],
                'per_class_metrics': class_metrics
            })

            # Save predictions
            pred_file = f"fe_{extractor_name}_{classifier_name}_iter{iteration}_fold{fold}.npz"
            np.savez(
                os.path.join(output_dir, pred_file),
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba
            )

        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'predictions_summary.csv'), index=False)

    # Save all results as pickle
    import pickle
    with open(os.path.join(output_dir, 'all_predictions.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n{'='*60}")
    print(f"Collected predictions from {len(all_results)} models")
    print(f"Results saved to: {output_dir}")

    # Print summary statistics
    if summary_df.empty:
        print("No results to summarize.")
        return

    print("\nSummary by Algorithm:")
    algo_summary = summary_df.groupby('algorithm').agg({
        'accuracy': ['mean', 'std'],
        'macro_avg_f1': ['mean', 'std']
    }).round(4)
    print(algo_summary)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Collect predictions from all trained models")
    parser.add_argument("--test-files", type=str, default="../res/test_files.txt",
                        help="Path to test files list")
    parser.add_argument("--results-dir", type=str, default="../results",
                        help="Directory containing all results")
    parser.add_argument("--output-dir", type=str, default="../all_predictions",
                        help="Directory to save collected predictions")

    args = parser.parse_args()

    # Convert to absolute paths
    test_files = os.path.abspath(args.test_files)
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Test files: {test_files}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Check if paths exist
    if not os.path.exists(test_files):
        print(f"Error: Test files not found at {test_files}")
        return

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found at {results_dir}")
        return

    # Run collection
    collect_all_predictions(test_files, results_dir, output_dir)


if __name__ == "__main__":
    main()