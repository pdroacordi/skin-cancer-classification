import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from typing import Tuple, Optional, Callable, Union, List

from config import cfg


def load_paths_labels(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image paths and labels from a text file.

    Args:
        file_path: Path to the file containing image paths and labels.

    Returns:
        tuple: (paths, labels) as numpy arrays
    """
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


def resize_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Resize an image to the target size.

    Args:
        image: Input image in BGR or RGB format
        target_size: Desired size as (height, width). If None, uses cfg.img_size

    Returns:
        Resized image
    """
    if target_size is None:
        target_size = cfg.img_size[:2]

    return cv2.resize(image, (target_size[1], target_size[0]))


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from a given path.

    Args:
        image_path: Path to the image file

    Returns:
        Image as BGR array or None if loading fails
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
    return image


def _get_preprocess_fn(model_name: str):
    """
    Return the backbone-specific preprocessing function for *model_name*.

    Resolved once per model name and cached via the module-level dict below.
    """
    if model_name == "Inception":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif model_name == "Xception":
        from tensorflow.keras.applications.xception import preprocess_input
    elif model_name == "ConvNeXt":
        from tensorflow.keras.applications.convnext import preprocess_input
    elif model_name == "EfficientNetV2S":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return preprocess_input


# Module-level cache: avoids re-importing on every image.
_PREPROCESS_FN_CACHE = {}


def apply_model_preprocessing(image: np.ndarray, model_name: str) -> np.ndarray:
    """
    Apply model-specific preprocessing to an image.

    Args:
        image: RGB image
        model_name: Name of the CNN model

    Returns:
        Preprocessed image
    """
    if model_name not in _PREPROCESS_FN_CACHE:
        _PREPROCESS_FN_CACHE[model_name] = _get_preprocess_fn(model_name)
    return _PREPROCESS_FN_CACHE[model_name](image)


def _apply_cutmix(images: np.ndarray, labels: np.ndarray,
                  alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    CutMix (Yun et al., 2019, ICCV).

    Samples a rectangular box of area ~ (1 - lam) and pastes it from a
    permuted batch member; labels are convex-combined proportional to the
    *actual* pasted-box area ratio.
    """
    n, H, W = images.shape[:3]
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = np.random.permutation(n)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    if y2 > y1 and x2 > x1:
        images = images.copy()
        images[:, y1:y2, x1:x2, :] = images[perm, y1:y2, x1:x2, :]
        lam_actual = 1.0 - ((y2 - y1) * (x2 - x1) / float(H * W))
    else:
        lam_actual = 1.0

    labels = lam_actual * labels + (1.0 - lam_actual) * labels[perm]
    return images, labels


class MemoryEfficientDataGenerator:
    """
    Memory-efficient data generator that loads and processes images in batches.
    """

    def __init__(self,
                 paths: Union[List[str], np.ndarray],
                 labels: Union[List[int], np.ndarray],
                 batch_size: int,
                 model_name: str,
                 augment_fn: Optional[Callable] = None,
                 shuffle: bool = True):
        """
        Initialize the data generator.

        Args:
            paths: List of image paths
            labels: List of image labels
            batch_size: Batch size
            model_name: CNN model name for preprocessing
            augment_fn: Function for applying data augmentation
            shuffle: Whether to shuffle the data
        """
        self.paths = np.array(paths) if not isinstance(paths, np.ndarray) else paths
        self.labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        self.batch_size = batch_size
        self.model_name = model_name
        self.augment_fn = augment_fn
        self.shuffle = shuffle
        self.n = len(paths)
        self.indices = np.arange(self.n)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        """Create a new iterator."""
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the next batch."""
        if not hasattr(self, '_current_idx'):
            self._current_idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)

        if self._current_idx >= self.n:
            self._current_idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration

        # Get batch indices
        end_idx = min(self._current_idx + self.batch_size, self.n)
        batch_indices = self.indices[self._current_idx:end_idx]
        self._current_idx += self.batch_size

        # Load and process batch
        batch_images = []
        batch_labels = []

        for i in batch_indices:
            image = load_image(self.paths[i])
            if image is None:
                continue

            image = resize_image(image, cfg.img_size[:2])

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply augmentation if available
            if self.augment_fn:
                image = self.augment_fn(image)

            # Apply model-specific preprocessing
            image = apply_model_preprocessing(image, self.model_name)

            batch_images.append(image)
            batch_labels.append(self.labels[i])

        if not batch_images:
            return self.__next__()

        batch_images = np.array(batch_images, dtype=np.float32)
        batch_labels = to_categorical(batch_labels, cfg.num_classes)

        # Mixup (Zhang et al., 2018) and CutMix (Yun et al., 2019).
        # When both flags are on, each batch randomly picks one 50/50.
        if self.augment_fn is not None and len(batch_images) > 1:
            use_mix = cfg.use_mixup
            use_cut = cfg.use_cutmix
            if use_mix and use_cut:
                if np.random.rand() < 0.5:
                    use_cut = False
                else:
                    use_mix = False
            if use_mix:
                lam = np.random.beta(0.4, 0.4)
                perm = np.random.permutation(len(batch_images))
                batch_images = lam * batch_images + (1 - lam) * batch_images[perm]
                batch_labels = lam * batch_labels + (1 - lam) * batch_labels[perm]
            elif use_cut:
                batch_images, batch_labels = _apply_cutmix(
                    batch_images, batch_labels, alpha=cfg.cutmix_alpha
                )

        return batch_images, batch_labels

    def get_keras_generator(self):
        """
        Create a generator that can be used with Keras model.fit().

        Returns:
            generator: A generator yielding (batch_images, batch_labels)
        """
        def generator():
            while True:
                try:
                    yield next(self)
                except StopIteration:
                    self._current_idx = 0
                    if self.shuffle:
                        np.random.shuffle(self.indices)
                    yield next(self)

        return generator()
