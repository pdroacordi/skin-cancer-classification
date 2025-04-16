"""
Data augmentation strategies for skin cancer image classification.
Provides various augmentation pipelines for different training scenarios.
"""

import sys

import albumentations as A
import numpy as np
from sklearn.utils import resample

sys.path.append('..')


class AugmentationFactory:
    """Factory class to create different augmentation pipelines."""

    @staticmethod
    def get_light_augmentation():
        """
        Light augmentation pipeline suitable for both training and feature extraction.

        Returns:
            A.Compose: Albumentations composition of transformations.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

    @staticmethod
    def get_medium_augmentation():
        """
        Medium augmentation pipeline for training.

        Returns:
            A.Compose: Albumentations composition of transformations.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.2),
            A.CLAHE(clip_limit=4.0, p=0.3),
        ])

    @staticmethod
    def get_strong_augmentation():
        """
        Strong augmentation pipeline for robust training.

        Returns:
            A.Compose: Albumentations composition of transformations.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.Transpose(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.3
            ),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
        ])

    @staticmethod
    def get_feature_extraction_augmentation():
        """
        Specialized augmentation pipeline for feature extraction.
        Creates multiple versions of each image with controlled transformations.

        Returns:
            list: List of augmentation pipelines to apply separately.
        """
        # Create several distinct augmentation pipelines
        # Each will be applied separately to create multiple augmented versions
        pipelines = [
            # Original image (no augmentation)
            A.Compose([]),

            # Rotation variants
            A.Compose([A.Rotate(limit=15, p=1.0)]),
            A.Compose([A.Rotate(limit=30, p=1.0)]),

            # Flip variants
            A.Compose([A.HorizontalFlip(p=1.0)]),
            A.Compose([A.VerticalFlip(p=1.0)]),

            # Brightness/contrast variants
            A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),

            # Slight zoom variants
            A.Compose([A.Affine(scale=1.05, p=1.0)]),
            A.Compose([A.Affine(scale=0.95, p=1.0)])
        ]

        return pipelines

def apply_augmentation(image, augmentation):
    """
    Apply an albumentations augmentation to an image.

    Args:
        image (numpy.array): Input image (RGB format).
        augmentation: Albumentations augmentation object.

    Returns:
        numpy.array: Augmented image.
    """
    # Apply augmentation
    augmented = augmentation(image=image)
    return augmented['image']

def generate_augmented_batch(images, labels, augmentation, batch_size=None):
    """
    Generate an augmented batch of images and corresponding labels.

    Args:
        images (numpy.array): Batch of images.
        labels (numpy.array): Corresponding labels.
        augmentation: Albumentations augmentation object.
        batch_size (int, optional): Size of output batch. If None, uses input size.

    Returns:
        tuple: (augmented_images, augmented_labels)
    """
    if batch_size is None:
        batch_size = len(images)

    # Randomly select images with replacement
    indices = np.random.choice(len(images), size=batch_size, replace=True)
    selected_images = images[indices]
    selected_labels = labels[indices]

    # Apply augmentation to each image
    augmented_images = np.array([
        apply_augmentation(img, augmentation) for img in selected_images
    ])

    return augmented_images, selected_labels

def apply_all_feature_augmentations(image, augmentation_list):
    """
    Apply all augmentations in a list to an image, returning multiple versions.

    Args:
        image (numpy.array): Input image (RGB format).
        augmentation_list (list): List of augmentation pipelines.

    Returns:
        list: List of augmented images.
    """
    augmented_images = []

    for aug in augmentation_list:
        augmented = apply_augmentation(image, aug)
        augmented_images.append(augmented)

    return augmented_images

def create_balanced_dataset(images, labels, target_count=None, augmentation=None):
    """
    Create a balanced dataset by augmenting underrepresented classes.

    Args:
        images (numpy.array): Images array.
        labels (numpy.array): Labels array.
        target_count (int, optional): Target count for each class. If None, uses the largest class count.
        augmentation: Albumentations augmentation object.

    Returns:
        tuple: (balanced_images, balanced_labels)
    """
    # Convert one-hot labels to class indices if needed
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        label_indices = np.argmax(labels, axis=1)
    else:
        label_indices = labels.copy()

    # Count samples in each class
    unique_classes = np.unique(label_indices)
    class_counts = {cls: np.sum(label_indices == cls) for cls in unique_classes}

    # Set target count to the largest class if not specified
    if target_count is None:
        target_count = max(class_counts.values())

    balanced_images = []
    balanced_labels = []

    # Process each class
    for cls in unique_classes:
        # Get indices for this class
        cls_indices = np.where(label_indices == cls)[0]
        cls_images = images[cls_indices]
        cls_labels = labels[cls_indices] if len(labels.shape) > 1 else labels[cls_indices]

        # If class already has enough samples, just take the original samples
        if len(cls_indices) >= target_count:
            balanced_images.append(cls_images[:target_count])
            balanced_labels.append(cls_labels[:target_count])
        else:
            # Need to augment this class
            # First include all original samples
            aug_images = list(cls_images)
            aug_labels = list(cls_labels)

            # Then add augmented samples until reaching target count
            samples_needed = target_count - len(cls_indices)

            if augmentation is not None:
                # Use provided augmentation
                while len(aug_images) < target_count:
                    # Randomly select an image to augment
                    idx = np.random.randint(0, len(cls_images))
                    img = cls_images[idx]
                    lbl = cls_labels[idx]

                    # Apply augmentation
                    aug_img = apply_augmentation(img, augmentation)

                    aug_images.append(aug_img)
                    aug_labels.append(lbl)
            else:
                # Use random oversampling without augmentation
                resampled_indices = resample(
                    cls_indices,
                    replace=True,
                    n_samples=samples_needed,
                    random_state=42
                )
                aug_images.extend(images[resampled_indices])
                aug_labels.extend(labels[resampled_indices])

            balanced_images.append(np.array(aug_images[:target_count]))
            balanced_labels.append(np.array(aug_labels[:target_count]))

    # Concatenate all classes
    balanced_images = np.concatenate(balanced_images, axis=0)
    balanced_labels = np.concatenate(balanced_labels, axis=0)

    # Shuffle the balanced dataset
    shuffle_indices = np.random.permutation(len(balanced_images))
    balanced_images = balanced_images[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]

    return balanced_images, balanced_labels