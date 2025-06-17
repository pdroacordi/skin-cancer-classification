from pathlib import Path
import tensorflow as tf, tensorflow_addons as tfa
import cv2
import numpy as np
from preprocessing.graphic.hair_removal.config import HairRemovalConfig


class HairRemovalDataset:
    """Carrega pasta hair/{dermoscopic_image,hair_mask} em tf.data.Dataset."""

    def __init__(self, root: Path, cfg: HairRemovalConfig):
        self.img_dir = root / "dermoscopic_image"
        self.mask_dir = root / "hair_mask"
        self.cfg = cfg
        self.img_paths = sorted(self.img_dir.glob("*"))
        self.mask_paths = sorted(self.mask_dir.glob("*"))
        assert len(self.img_paths) == len(self.mask_paths), "imgs ≠ masks"

    # ---------- util internos ----------
    def _load_pair_np(self, img_p: tf.Tensor, mask_p: tf.Tensor):
        """Load image and mask pair. Handles tensor to string conversion."""
        # Convert tensor to string
        img_path = img_p.numpy().decode('utf-8')
        mask_path = mask_p.numpy().decode('utf-8')

        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        img = img.astype(np.float32) / 255.0

        # Load and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = cv2.resize(mask, (self.cfg.img_size, self.cfg.img_size))
        mask = (mask > 127).astype(np.float32)[..., None]

        return img, mask

    def _map_tf(self, ip, mp):
        """TensorFlow mapping function with augmentation."""
        # Load image and mask using py_function
        img, mask = tf.py_function(
            func=self._load_pair_np,
            inp=[ip, mp],
            Tout=[tf.float32, tf.float32]
        )

        # Set shapes (required after py_function)
        img.set_shape([self.cfg.img_size, self.cfg.img_size, 3])
        mask.set_shape([self.cfg.img_size, self.cfg.img_size, 1])

        # Apply the same augmentation to both image and mask
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)

        # Random vertical flip
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)

        return img, mask

    # ---------- API pública ----------
    def tf_datasets(self):
        """Create train and validation datasets."""
        # Convert paths to strings
        img_paths_str = [str(p) for p in self.img_paths]
        mask_paths_str = [str(p) for p in self.mask_paths]

        # Create dataset from file paths
        ds = tf.data.Dataset.from_tensor_slices((img_paths_str, mask_paths_str))

        # Shuffle before splitting
        ds = ds.shuffle(buffer_size=len(self.img_paths), seed=42)

        # Split into train and validation
        val_size = int(0.1 * len(self.img_paths))
        ds_val = ds.take(val_size)
        ds_train = ds.skip(val_size)

        # Apply mapping, batching, and prefetching to validation set
        ds_val = (ds_val
                  .map(self._map_tf, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(self.cfg.batch_size)
                  .prefetch(tf.data.AUTOTUNE))

        # Apply mapping, batching, prefetching, and repeat to training set
        ds_train = (ds_train
                    .map(self._map_tf, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(self.cfg.batch_size)
                    .prefetch(tf.data.AUTOTUNE))

        return ds_train, ds_val

    def __len__(self):
        """Return number of samples."""
        return len(self.img_paths)


# Alternative implementation using tf.keras.utils.load_img for better compatibility
class HairRemovalDatasetKeras:
    """Alternative dataset implementation using Keras utilities."""

    def __init__(self, root: Path, cfg: HairRemovalConfig):
        self.img_dir = root / "dermoscopic_image"
        self.mask_dir = root / "hair_mask"
        self.cfg = cfg
        self.img_paths = sorted(self.img_dir.glob("*"))
        self.mask_paths = sorted(self.mask_dir.glob("*"))
        assert len(self.img_paths) == len(self.mask_paths), "imgs ≠ masks"

    def _load_and_preprocess_image(self, path):
        """Load and preprocess image using TensorFlow ops."""
        # Read file
        image = tf.io.read_file(path)
        # Decode image
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        # Ensure shape is set
        image.set_shape([None, None, 3])
        # Resize
        image = tf.image.resize(image, [self.cfg.img_size, self.cfg.img_size])
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def _load_and_preprocess_mask(self, path):
        """Load and preprocess mask using TensorFlow ops."""
        # Read file
        mask = tf.io.read_file(path)
        # Decode as grayscale
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        # Ensure shape is set
        mask.set_shape([None, None, 1])
        # Resize
        mask = tf.image.resize(mask, [self.cfg.img_size, self.cfg.img_size])
        # Threshold to binary
        mask = tf.cast(mask > 127, tf.float32)
        return mask

    def _augment(self, image, mask):
        # 1) Random flips
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask  = tf.image.flip_left_right(mask)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask  = tf.image.flip_up_down(mask)

        # 2) Small rotation ±10°
        angle = tf.random.uniform((), -0.17, 0.17)
        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
        mask  = tfa.image.rotate(mask,  angle, interpolation='NEAREST')

        # 3) Random zoom [0.95–1.05]
        h, w = self.cfg.img_size, self.cfg.img_size
        scale = tf.random.uniform((), 0.95, 1.05)
        new_h = tf.cast(h * scale, tf.int32)
        new_w = tf.cast(w * scale, tf.int32)
        image = tf.image.resize(image, [new_h, new_w])
        mask  = tf.image.resize(mask,  [new_h, new_w], method='nearest')
        image = tf.image.resize_with_crop_or_pad(image, h, w)
        mask  = tf.image.resize_with_crop_or_pad(mask,  h, w)

        # 4) Color jitter + light noise
        image = tf.image.random_hue(image, 0.05)
        image = tf.image.random_saturation(image, 0.9, 1.1)
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)

        # 5) Photometric jitter + optional blur
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        if tf.random.uniform(()) > 0.5:
            image = tfa.image.gaussian_filter2d(
                image,
                filter_shape=(3,3),
                sigma=1.0
            )

        return image, mask


    def tf_datasets(self):
        """Create train and validation datasets using TensorFlow ops."""
        # Convert paths to strings
        img_paths_str = [str(p) for p in self.img_paths]
        mask_paths_str = [str(p) for p in self.mask_paths]

        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((img_paths_str, mask_paths_str))

        # Shuffle
        ds = ds.shuffle(buffer_size=len(self.img_paths), seed=42)

        # Split
        val_size = int(0.1 * len(self.img_paths))
        ds_val = ds.take(val_size)
        ds_train = ds.skip(val_size)

        # Process validation set (no augmentation)
        ds_val = (ds_val
                  .map(lambda ip, mp: (self._load_and_preprocess_image(ip),
                                       self._load_and_preprocess_mask(mp)),
                       num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(self.cfg.batch_size)
                  .prefetch(tf.data.AUTOTUNE))

        # Process training set (with augmentation)
        ds_train = (ds_train
                    .map(lambda ip, mp: (self._load_and_preprocess_image(ip),
                                         self._load_and_preprocess_mask(mp)),
                         num_parallel_calls=tf.data.AUTOTUNE)
                    .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(self.cfg.batch_size)
                    .prefetch(tf.data.AUTOTUNE))

        return ds_train, ds_val

    def __len__(self):
        return len(self.img_paths)