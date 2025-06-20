from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing.graphic.hair_removal.config import HairRemovalConfig


class HairRemovalDataset:
    """Enhanced dataset with improved augmentation strategies"""

    def __init__(self, root: Path, cfg: HairRemovalConfig):
        self.img_dir = root / "dermoscopic_image"
        self.mask_dir = root / "hair_mask"
        self.cfg = cfg
        self.img_paths = sorted(self.img_dir.glob("*"))
        self.mask_paths = sorted(self.mask_dir.glob("*"))
        assert len(self.img_paths) == len(self.mask_paths), "imgs ≠ masks"

        # Enable mixed precision if configured
        if cfg.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.img_paths)

    def _augment_pair(self, image, mask):
        """Apply synchronized augmentations to image and mask"""
        import tensorflow_addons as tfa

        # Random rotation
        if tf.random.uniform(()) < self.cfg.augmentation_prob:
            angle = tf.random.uniform((), -self.cfg.rotation_range * np.pi/180,
                                     self.cfg.rotation_range * np.pi/180)
            image = tfa.image.rotate(image, angle, interpolation='bilinear')
            mask = tfa.image.rotate(mask, angle, interpolation='nearest')

        # Random flips
        if tf.random.uniform(()) < 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if tf.random.uniform(()) < 0.3:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)

        # Random zoom
        if tf.random.uniform(()) < self.cfg.augmentation_prob:
            scale = tf.random.uniform((), self.cfg.zoom_range[0], self.cfg.zoom_range[1])
            new_size = tf.cast(tf.cast([self.cfg.img_size, self.cfg.img_size], tf.float32) * scale, tf.int32)

            image = tf.image.resize(image, new_size)
            mask = tf.image.resize(mask, new_size, method='nearest')

            # Crop or pad back to original size
            image = tf.image.resize_with_crop_or_pad(image, self.cfg.img_size, self.cfg.img_size)
            mask = tf.image.resize_with_crop_or_pad(mask, self.cfg.img_size, self.cfg.img_size)

        # Color augmentations (only for image)
        if tf.random.uniform(()) < self.cfg.augmentation_prob:
            image = tf.image.random_brightness(image,
                                             max_delta=(self.cfg.brightness_range[1] - 1))
            image = tf.image.random_contrast(image,
                                           self.cfg.contrast_range[0],
                                           self.cfg.contrast_range[1])
            image = tf.image.random_hue(image, 0.05)
            image = tf.image.random_saturation(image, 0.8, 1.2)

        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)

        # Add noise
        if tf.random.uniform(()) < 0.3:
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
            image = tf.clip_by_value(image + noise, 0.0, 1.0)

        return image, mask

    def _load_and_preprocess(self, img_path, mask_path):
        """Load and preprocess image-mask pair"""
        # Load image
        image = tf.io.read_file(img_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [self.cfg.img_size, self.cfg.img_size])
        image = tf.cast(image, tf.float32) / 255.0

        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask.set_shape([None, None, 1])
        mask = tf.image.resize(mask, [self.cfg.img_size, self.cfg.img_size], method='nearest')
        mask = tf.cast(mask > 127, tf.float32)

        return image, mask

    def tf_datasets(self, validation_split=0.15):
        """Create training and validation datasets with improved strategies"""
        # Convert paths to strings
        img_paths_str = [str(p) for p in self.img_paths]
        mask_paths_str = [str(p) for p in self.mask_paths]

        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((img_paths_str, mask_paths_str))

        # Shuffle with large buffer
        ds = ds.shuffle(buffer_size=len(self.img_paths), seed=42)

        # Split into train and validation
        val_size = int(validation_split * len(self.img_paths))
        ds_val = ds.take(val_size)
        ds_train = ds.skip(val_size)

        # Process validation set (no augmentation)
        ds_val = (ds_val
                 .map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(self.cfg.batch_size)
                 .prefetch(tf.data.AUTOTUNE))

        # Process training set (with augmentation)
        ds_train = (ds_train
                   .map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                   .map(self._augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(self.cfg.batch_size)
                   .repeat()  # Repeat for training
                   .prefetch(tf.data.AUTOTUNE))

        return ds_train, ds_val