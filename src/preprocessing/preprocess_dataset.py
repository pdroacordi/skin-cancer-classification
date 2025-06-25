"""
Preprocess entire dataset and save preprocessed images.
This script applies graphic preprocessing (hair removal, contrast enhancement, etc.)
to all images in the dataset and saves them to a new directory structure.

Usage:
    python preprocess_dataset.py --metadata path/to/metadata.csv \
                                --images-dir1 path/to/HAM10000_images_part_1 \
                                --images-dir2 path/to/HAM10000_images_part_2 \
                                --output-dir path/to/preprocessed_images \
                                --hair-removal --contrast-enhancement
"""

import argparse
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from .graphic.pipeline import apply_graphic_preprocessing


def process_single_image(args):
    """Process a single image with error handling."""
    image_path, output_path, use_hair_removal, use_contrast_enhancement, use_segmentation = args

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Skip if already processed
        if os.path.exists(output_path):
            return True, image_path, "Already processed"

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return False, image_path, "Failed to load image"

        # Apply preprocessing
        processed = apply_graphic_preprocessing(
            image,
            use_hair_removal=use_hair_removal,
            use_contrast_enhancement=use_contrast_enhancement,
            visualize=False
        )

        # Save processed image
        success = cv2.imwrite(output_path, processed)

        if success:
            return True, image_path, "Success"
        else:
            return False, image_path, "Failed to save image"

    except Exception as e:
        return False, image_path, str(e)


def preprocess_dataset(metadata_path, image_dirs, output_dir,
                       use_hair_removal=True, use_contrast_enhancement=True,
                       use_segmentation=False, n_workers=None):
    """
    Preprocess entire dataset in parallel.

    Args:
        metadata_path: Path to HAM10000 metadata CSV
        image_dirs: List of directories containing images
        output_dir: Output directory for preprocessed images
        use_hair_removal: Whether to apply hair removal
        use_contrast_enhancement: Whether to apply contrast enhancement
        use_segmentation: Whether to apply segmentation
        n_workers: Number of parallel workers (None for auto)
    """
    print("Loading metadata...")
    metadata = pd.read_csv(metadata_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of image paths
    image_tasks = []
    missing_count = 0

    for _, row in metadata.iterrows():
        image_id = row['image_id']
        image_filename = f"{image_id}.jpg"

        # Find image in provided directories
        image_path = None
        for img_dir in image_dirs:
            candidate_path = os.path.join(img_dir, image_filename)
            if os.path.exists(candidate_path):
                image_path = candidate_path
                break

        if image_path is None:
            missing_count += 1
            continue

        # Output path maintains the same filename
        output_path = output_dir / image_filename

        image_tasks.append((
            image_path,
            str(output_path),
            use_hair_removal,
            use_contrast_enhancement,
            use_segmentation
        ))

    print(f"Found {len(image_tasks)} images to process ({missing_count} missing)")

    # Process images in parallel
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Cap at 8 workers to avoid memory issues

    print(f"Processing images using {n_workers} workers...")

    success_count = 0
    error_count = 0
    errors = []

    with Pool(n_workers) as pool:
        with tqdm(total=len(image_tasks), desc="Processing images") as pbar:
            for success, image_path, message in pool.imap_unordered(process_single_image, image_tasks):
                pbar.update(1)
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append((image_path, message))

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")

    if errors:
        print("\nErrors encountered:")
        for path, error in errors[:10]:  # Show first 10 errors
            print(f"  {path}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # Save preprocessing info
    info_path = output_dir / "preprocessing_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Preprocessing Configuration\n")
        f.write(f"==========================\n")
        f.write(f"Hair Removal: {use_hair_removal}\n")
        f.write(f"Contrast Enhancement: {use_contrast_enhancement}\n")
        f.write(f"Segmentation: {use_segmentation}\n")
        f.write(f"\nImages processed: {success_count}\n")
        f.write(f"Images failed: {error_count}\n")
        f.write(f"Images missing: {missing_count}\n")

    print(f"\nPreprocessing info saved to: {info_path}")

    # Update split files to use preprocessed images
    update_split_files(output_dir)


def update_split_files(preprocessed_dir):
    """Update train/val/test split files to point to preprocessed images."""
    split_files = ['train_files.txt', 'val_files.txt', 'test_files.txt']
    res_dir = Path('./res')

    for split_file in split_files:
        original_path = res_dir / split_file
        if not original_path.exists():
            print(f"Warning: {original_path} not found")
            continue

        # Read original split file
        with open(original_path, 'r') as f:
            lines = f.readlines()

        # Create new split file with preprocessed paths
        new_path = res_dir / f"preprocessed_{split_file}"
        with open(new_path, 'w') as f:
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    original_image_path, label = parts
                    image_filename = os.path.basename(original_image_path)
                    new_image_path = os.path.join(preprocessed_dir, image_filename)
                    f.write(f"{new_image_path}\t{label}\n")

        print(f"Created {new_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess HAM10000 dataset")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to HAM10000 metadata CSV")
    parser.add_argument("--images-dir1", type=str, required=True,
                        help="First directory containing images")
    parser.add_argument("--images-dir2", type=str, required=True,
                        help="Second directory containing images")
    parser.add_argument("--output-dir", type=str, default="./data/preprocessed_images",
                        help="Output directory for preprocessed images")
    parser.add_argument("--hair-removal", action="store_true",
                        help="Apply hair removal")
    parser.add_argument("--contrast-enhancement", action="store_true",
                        help="Apply contrast enhancement")
    parser.add_argument("--segmentation", action="store_true",
                        help="Apply segmentation")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")

    args = parser.parse_args()

    # Check if hair removal model exists
    if args.hair_removal:
        from .graphic.hair_removal.config import HairRemovalConfig
        cfg = HairRemovalConfig()
        if not cfg.model_weights.exists():
            print(f"Error: Hair removal model not found at {cfg.model_weights}")
            print("Please train the hair removal model first or disable --hair-removal")
            sys.exit(1)

    image_dirs = [args.images_dir1, args.images_dir2]

    preprocess_dataset(
        metadata_path=args.metadata,
        image_dirs=image_dirs,
        output_dir=args.output_dir,
        use_hair_removal=args.hair_removal,
        use_contrast_enhancement=args.contrast_enhancement,
        use_segmentation=args.segmentation,
        n_workers=args.workers
    )


if __name__ == "__main__":
    main()