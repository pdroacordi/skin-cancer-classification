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


def setup_segmentation_data(training_input_dir, validation_input_dir,
                            training_ground_truth_dir, output_dir):
    """
    Setup segmentation training data using ISIC-2018 Task 1 dataset structure.

    Args:
        training_input_dir (str): Directory containing training images.
        validation_input_dir (str): Directory containing validation images.
        training_ground_truth_dir (str): Directory containing ground truth masks.
        output_dir (str): Output directory for processed segmentation data.
    """
    import cv2
    import numpy as np
    import os
    import glob

    print("Setting up segmentation training data from ISIC-2018 Task 1 dataset...")

    # Create directory structure
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    train_img_dir = os.path.join(train_dir, 'images')
    train_mask_dir = os.path.join(train_dir, 'masks')
    val_img_dir = os.path.join(val_dir, 'images')
    val_mask_dir = os.path.join(val_dir, 'masks')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    # Find all training image files
    training_images = sorted(glob.glob(os.path.join(training_input_dir, '*.jpg')))
    if not training_images:
        training_images = sorted(glob.glob(os.path.join(training_input_dir, '*.png')))

    # Extract image IDs from filenames
    training_ids = [os.path.splitext(os.path.basename(f))[0] for f in training_images]

    # Find corresponding masks for training images
    training_masks = {}
    for img_id in training_ids:
        # Check different possible mask naming patterns
        mask_patterns = [
            os.path.join(training_ground_truth_dir, f"{img_id}_segmentation.png"),
            os.path.join(training_ground_truth_dir, f"{img_id}_mask.png"),
            os.path.join(training_ground_truth_dir, f"{img_id}.png")
        ]

        for pattern in mask_patterns:
            if os.path.exists(pattern):
                training_masks[img_id] = pattern
                break

    # Find all validation image files
    validation_images = sorted(glob.glob(os.path.join(validation_input_dir, '*.jpg')))
    if not validation_images:
        validation_images = sorted(glob.glob(os.path.join(validation_input_dir, '*.png')))

    # Extract validation image IDs
    validation_ids = [os.path.splitext(os.path.basename(f))[0] for f in validation_images]

    # Check if we found images and masks
    valid_training_ids = [img_id for img_id in training_ids if img_id in training_masks]

    if not valid_training_ids:
        raise ValueError(f"No matching mask files found in {training_ground_truth_dir}")

    print(f"Found {len(valid_training_ids)} valid training image-mask pairs")
    print(f"Found {len(validation_ids)} validation images")

    # Process training images
    print("Processing training images...")
    for idx, img_id in enumerate(valid_training_ids):
        # Get image and mask paths
        img_path = next(path for path in training_images if img_id in path)
        mask_path = training_masks[img_id]

        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        img = cv2.resize(img, (299, 299))

        # Load and resize mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error loading mask: {mask_path}")
            continue

        mask = cv2.resize(mask, (299, 299), interpolation=cv2.INTER_NEAREST)

        # Ensure mask is binary (0 and 255)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Save processed image and mask
        output_img_path = os.path.join(train_img_dir, f"{img_id}.jpg")
        output_mask_path = os.path.join(train_mask_dir, f"{img_id}.png")

        cv2.imwrite(output_img_path, img)
        cv2.imwrite(output_mask_path, binary_mask)

        # Print progress periodically
        if (idx + 1) % 100 == 0 or (idx + 1) == len(valid_training_ids):
            print(f"Processed {idx + 1}/{len(valid_training_ids)} training images")

    # Process validation images (note: we don't need masks for validation as they'll be generated)
    print("Processing validation images...")
    for idx, img_id in enumerate(validation_ids):
        # Get image path
        img_path = next(path for path in validation_images if img_id in path)

        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        img = cv2.resize(img, (299, 299))

        # Save processed image
        output_img_path = os.path.join(val_img_dir, f"{img_id}.jpg")
        cv2.imwrite(output_img_path, img)

        # Print progress periodically
        if (idx + 1) % 100 == 0 or (idx + 1) == len(validation_ids):
            print(f"Processed {idx + 1}/{len(validation_ids)} validation images")

    print(f"Segmentation data setup complete. Data saved to {output_dir}")
    return train_dir, val_dir

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

    # Train UNet segmentator
    parser.add_argument("--setup-segmentation", action="store_true",
                        help="Prepare data for segmentation model training. Requires --isic-training-input, --isic-validation-input, and --isic-training-ground-truth.")

    parser.add_argument("--train-segmentation", action="store_true",
                        help="Train the segmentation model for lesion segmentation.")

    parser.add_argument("--segmentation-dir", type=str, default="../data/segmentation_data",
                        help="Directory to store prepared segmentation data.")

    parser.add_argument("--isic-training-input", type=str,
                        help="Path to the ISIC training input images.")

    parser.add_argument("--isic-validation-input", type=str,
                        help="Path to the ISIC validation input images.")

    parser.add_argument("--isic-training-ground-truth", type=str,
                        help="Path to the ISIC training ground truth masks.")

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

    # Setup segmentation data if requested
    if args.setup_segmentation:
        if not (args.isic_training_input and args.isic_validation_input and args.isic_training_ground_truth):
            parser.error(
                "--setup-segmentation requires --isic-training-input, --isic-validation-input, and --isic-training-ground-truth")

        train_dir, val_dir = setup_segmentation_data(
            training_input_dir=args.isic_training_input,
            validation_input_dir=args.isic_validation_input,
            training_ground_truth_dir=args.isic_training_ground_truth,
            output_dir=args.segmentation_dir
        )
    else:
        # Default paths if setup not requested
        train_dir = os.path.join(args.segmentation_dir, 'train')
        val_dir = os.path.join(args.segmentation_dir, 'val')

    # Train segmentation model if requested
    if args.train_segmentation:
        from utils.graphic_preprocessing import train_segmentation_model

        print("\n" + "=" * 50)
        print("Training Segmentation Model for Lesion Boundary Detection")
        print("=" * 50)

        # Create directories for results if they don't exist
        unet_dir = "./results/unet_segmentation_model"
        os.makedirs(os.path.join(unet_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(unet_dir, "results"), exist_ok=True)

        # Train the segmentation model
        train_segmentation_model(
            train_data_path=train_dir,
            val_data_path=val_dir,
            epochs=100,
            batch_size=8,
            save_path=unet_dir
        )

        print("\nSegmentation model training completed")

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
            class_names=class_names
        )

        results["cnn"] = cnn_results

        # Clear memory
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