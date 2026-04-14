"""
Data augmentation strategies for skin cancer image classification.
Provides various augmentation pipelines for different training scenarios.
"""

import albumentations as A
import numpy as np
from tensorflow.keras.utils import Sequence


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
            A.CoarseDropout(p=0.2),
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

    class AugmentedDataGenerator(Sequence):
        def __init__(self, x_set, y_set, batch_size, augmentation):
            self.x = x_set
            self.y = y_set
            self.batch_size = batch_size
            self.augmentation = augmentation
            self.indices = np.arange(len(self.x))
            np.random.shuffle(self.indices)

        def __len__(self):
            return int(np.ceil(len(self.x) / self.batch_size))

        def __getitem__(self, idx):
            from config import IMG_SIZE  # lazy import — IMG_SIZE may be overridden by CLI at runtime

            # Get batch indices
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = self.x[batch_indices]
            batch_y = self.y[batch_indices]

            expected_shape = IMG_SIZE  # e.g. (380, 380, 3) for EfficientNet, (299, 299, 3) for Inception

            # Apply augmentation
            augmented_batch = []
            for image in batch_x:
                if image.shape != expected_shape:
                    print(f"Skipping augmentation: image shape {image.shape} != expected {expected_shape}")
                    augmented_batch.append(image)
                    continue

                try:
                    # Convert to uint8 if needed (albumentations expects uint8)
                    if image.dtype != np.uint8:
                        if image.min() < 0 or image.max() > 1:
                            image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                        else:
                            image_uint8 = (image * 255).astype(np.uint8)
                    else:
                        image_uint8 = image

                    aug_image = self.augmentation(image=image_uint8)['image']

                    # Convert back to the original dtype and range
                    if image.dtype != np.uint8:
                        if image.min() < 0 or image.max() > 1:
                            aug_image = (aug_image / 255.0) * (image.max() - image.min()) + image.min()
                            aug_image = aug_image.astype(image.dtype)
                        else:
                            aug_image = (aug_image / 255.0).astype(image.dtype)

                    augmented_batch.append(aug_image)
                except Exception as e:
                    print(f"Augmentation error: {e}. Using original image.")
                    augmented_batch.append(image)

            return np.array(augmented_batch), batch_y

        def on_epoch_end(self):
            # Shuffle indices at the end of each epoch
            np.random.shuffle(self.indices)

