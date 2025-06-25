import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from typing import Tuple, Optional, Callable, Union, List
import sys

sys.path.append('..')
from config import NUM_CLASSES, IMG_SIZE


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
        target_size: Desired size as (height, width). If None, uses IMG_SIZE

    Returns:
        Resized image
    """
    if target_size is None:
        target_size = IMG_SIZE[:2]

    resized_image = cv2.resize(image, (target_size[1], target_size[0]))
    return resized_image


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


def apply_model_preprocessing(image: np.ndarray, model_name: str) -> np.ndarray:
    """
    Apply model-specific preprocessing to an image.

    Args:
        image: RGB image
        model_name: Name of the CNN model

    Returns:
        Preprocessed image
    """
    from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
    from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
    from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception

    if model_name == "VGG19":
        return preprocess_input_vgg19(image.copy())
    elif model_name == "Inception":
        return preprocess_input_inception(image.copy())
    elif model_name == "ResNet":
        return preprocess_input_resnet(image.copy())
    elif model_name == "Xception":
        return preprocess_input_xception(image.copy())
    else:
        raise ValueError(f"Unsupported model: {model_name}")


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

            image = resize_image(image, IMG_SIZE[:2])

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
        batch_labels = to_categorical(batch_labels, NUM_CLASSES)

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


def load_and_preprocess_dataset(paths: Union[List[str], np.ndarray],
                                labels: Union[List[int], np.ndarray],
                                model_name: str,
                                augment_fn: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a full dataset into memory.

    Args:
        paths: List of image paths
        labels: List of image labels
        model_name: CNN model name for preprocessing
        augment_fn: Function for applying data augmentation

    Returns:
        tuple: (preprocessed_images, one_hot_labels)
    """
    images = []
    for path in paths:
        image = load_image(path)
        if image is None:
            continue

        image = resize_image(image, IMG_SIZE[:2])

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation if available
        if augment_fn:
            image = augment_fn(image)

        # Apply model-specific preprocessing
        image = apply_model_preprocessing(image, model_name)

        images.append(image)

    images = np.array(images, dtype=np.float32)
    labels_one_hot = to_categorical(labels, NUM_CLASSES)

    return images, labels_one_hot