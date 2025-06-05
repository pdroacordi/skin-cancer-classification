"""
Main entry point for skin cancer classification project.
Handles command-line arguments and runs the appropriate pipeline.
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_FILES_PATH,
    VAL_FILES_PATH,
    TEST_FILES_PATH,
    CNN_MODEL,
    CLASSICAL_CLASSIFIER_MODEL,
    USE_FINE_TUNING,
    USE_GRAPHIC_PREPROCESSING,
    USE_DATA_AUGMENTATION,
    NUM_KFOLDS, BATCH_SIZE
)
from pipelines.cnn_classifier import run_cnn_classifier_pipeline
from pipelines.feature_extraction import run_feature_extraction_pipeline


def setup_environment():
    """Configure the environment for the experiment."""
    # Set up GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found. Running on CPU.")

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

def create_dataset_splits(metadata_path, image_dir_1, image_dir_2, output_dir='./res'):
    """
    Create train/val/test splits from the metadata and save them to files.

    Args:
        metadata_path (str): Path to the HAM10000 metadata CSV.
        image_dir_1 (str): First directory containing images.
        image_dir_2 (str): Second directory containing images.
        output_dir (str): Directory to save the split files.
    """
    print("Creating dataset splits...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the metadata
    metadata = pd.read_csv(metadata_path)

    # Add file extension to image IDs
    metadata['image_file'] = metadata['image_id'] + ".jpg"

    # Map to full path
    metadata['image_path'] = metadata['image_file'].apply(
        lambda x: os.path.join(image_dir_1, x) if os.path.exists(os.path.join(image_dir_1, x))
        else os.path.join(image_dir_2, x)
    )

    # Verify that all images exist
    missing_images = metadata[~metadata['image_path'].apply(os.path.exists)]
    if not missing_images.empty:
        print(f"Warning: {len(missing_images)} images not found:")
        print(missing_images['image_id'].values[:5])
        print("..." if len(missing_images) > 5 else "")
        metadata = metadata[metadata['image_path'].apply(os.path.exists)]

    # Map class names to integer labels
    class_names = sorted(metadata['dx'].unique())
    class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    metadata['label'] = metadata['dx'].map(class_to_label)

    # Split into train+val and test
    train_val_metadata, test_metadata = train_test_split(
        metadata, test_size=0.15, random_state=42, stratify=metadata['dx']
    )

    # Split train+val into train and validation
    train_metadata, val_metadata = train_test_split(
        train_val_metadata, test_size=0.15, random_state=42, stratify=train_val_metadata['dx']
    )

    # Save splits to files
    train_metadata[['image_path', 'label']].to_csv(
        os.path.join(output_dir, "train_files.txt"), index=False, header=False, sep='\t'
    )

    val_metadata[['image_path', 'label']].to_csv(
        os.path.join(output_dir, "val_files.txt"), index=False, header=False, sep='\t'
    )

    test_metadata[['image_path', 'label']].to_csv(
        os.path.join(output_dir, "test_files.txt"), index=False, header=False, sep='\t'
    )

    # Save class names and their indices
    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        for cls_name, idx in class_to_label.items():
            f.write(f"{idx}\t{cls_name}\n")

    # Print statistics
    print("Dataset splits created:")
    print(f"  Training set: {len(train_metadata)} images")
    print(f"  Validation set: {len(val_metadata)} images")
    print(f"  Test set: {len(test_metadata)} images")
    print(f"  Total: {len(metadata)} images")

    # Class distribution
    print("\nClass distribution:")
    for cls_name in class_names:
        train_count = sum(train_metadata['dx'] == cls_name)
        val_count = sum(val_metadata['dx'] == cls_name)
        test_count = sum(test_metadata['dx'] == cls_name)
        total_count = sum(metadata['dx'] == cls_name)

        print(f"  {cls_name}: {train_count} train, {val_count} val, {test_count} test, {total_count} total")

    return class_to_label


def print_experiment_configuration():
    """Print the current experiment configuration."""
    print("\nExperiment Configuration:")
    print(f"  CNN Model: {CNN_MODEL}")
    print(f"  Classical Classifier: {CLASSICAL_CLASSIFIER_MODEL}")
    print(f"  Use Fine-tuning: {USE_FINE_TUNING}")
    print(f"  Use Graphics Preprocessing: {USE_GRAPHIC_PREPROCESSING}")
    print(f"  Use Data Augmentation: {USE_DATA_AUGMENTATION}")
    print(f"  K-Fold Cross-validation: {NUM_KFOLDS > 1}")


def get_class_names(class_names_path):
    """
    Get class names from a file.

    Args:
        class_names_path (str): Path to file containing class names.

    Returns:
        list: List of class names.
    """
    class_names = []

    if not os.path.exists(class_names_path):
        return None

    try:
        with open(class_names_path, 'r') as f:
            for line in f:
                idx, name = line.strip().split('\t')
                class_names.append(name)
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None

    return class_names


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Skin Cancer Classification")

    # Data preparation
    parser.add_argument("--create-splits", action="store_true",
                        help="Create new dataset splits")
    parser.add_argument("--metadata", type=str,
                        help="Path to HAM10000 metadata CSV")
    parser.add_argument("--images-dir1", type=str,
                        help="First directory containing images")
    parser.add_argument("--images-dir2", type=str,
                        help="Second directory containing images")

    # Pipeline selection
    parser.add_argument("--pipeline", type=str, choices=["cnn", "feature-extraction", "both"],
                        default="both", help="Pipeline to run")

    # Cross-validation
    parser.add_argument("--cv", action="store_true",
                        help="Run cross-validation instead of fixed splits")

    # Data paths
    parser.add_argument("--train-files", type=str, default=TRAIN_FILES_PATH,
                        help="Path to training files list")
    parser.add_argument("--val-files", type=str, default=VAL_FILES_PATH,
                        help="Path to validation files list")
    parser.add_argument("--test-files", type=str, default=TEST_FILES_PATH,
                        help="Path to test files list")

    parser.add_argument("--train-segmentation", action="store_true",
                        help="Train the segmentation model for lesion segmentation.")

    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training.")

    args = parser.parse_args()

    # Setup environment
    setup_environment()

    # Create dataset splits if requested
    if args.create_splits:
        if not (args.metadata and args.images_dir1 and args.images_dir2):
            parser.error("--create-splits requires --metadata, --images-dir1, and --images-dir2")

        class_to_label = create_dataset_splits(
            metadata_path=args.metadata,
            image_dir_1=args.images_dir1,
            image_dir_2=args.images_dir2
        )

    # Get class names
    class_names_path = os.path.join(os.path.dirname(args.train_files), "class_names.txt")
    class_names = get_class_names(class_names_path)

    if class_names:
        print("\nClass Names:")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")

    # Print experiment configuration
    print_experiment_configuration()

    # Run the selected pipeline(s)
    results = {}

    if args.pipeline in ["cnn", "both"]:
        print("\n" + "=" * 50)
        print("Running CNN Classifier Pipeline")
        print("=" * 50)

        cnn_results = run_cnn_classifier_pipeline(
            train_files_path=args.train_files,
            val_files_path=args.val_files,
            test_files_path=args.test_files,
            run_kfold=args.cv,
            class_names=class_names,
            skip_training=args.skip_train
        )

        results["cnn"] = cnn_results

        gc.collect()

    if args.pipeline in ["feature-extraction", "both"]:
        print("\n" + "=" * 50)
        print("Running Feature Extraction + Classical ML Pipeline")
        print("=" * 50)

        fe_results = run_feature_extraction_pipeline(
            train_files_path=args.train_files,
            val_files_path=args.val_files,
            test_files_path=args.test_files,
            use_kfold=args.cv,
            fine_tune_extractor=USE_FINE_TUNING,
            balance_features=USE_DATA_AUGMENTATION,
            tune_hyperparams=False,
            class_names=class_names
        )

        results["feature_extraction"] = fe_results

    print("\n" + "=" * 50)
    print("Experiment completed successfully!")
    print("=" * 50)

    return results


if __name__ == "__main__":
    main()