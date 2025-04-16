import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

import sys
sys.path.append('..')
from config import NUM_CLASSES, IMG_SIZE


def load_paths_labels(file_path):
    """
    Load image paths and labels from a text file.

    Args:
        file_path (str): Path to the file containing image paths and labels.

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


def resize_image(image, target_size=None):
    """
    Resize an image to the target size.

    Args:
        image (numpy.array): Input image in BGR or RGB format.
        target_size (tuple): Desired size as (height, width). If None, uses IMG_SIZE.

    Returns:
        numpy.array: Resized image.
    """
    if target_size is None:
        target_size = IMG_SIZE[:2]

    resized_image = cv2.resize(image, (target_size[1], target_size[0]))
    return resized_image


def load_image(image_path):
    """
    Load an image from a given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.array: Image as BGR array or None if loading fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
    return image


def apply_model_preprocessing(image, model_name):
    """
    Apply model-specific preprocessing to an image.

    Args:
        image (numpy.array): RGB image.
        model_name (str): Name of the CNN model.

    Returns:
        numpy.array: Preprocessed image.
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

    def __init__(self, paths, labels, batch_size, model_name, preprocess_fn=None,
                 augment_fn=None, shuffle=True):
        """
        Initialize the data generator.

        Args:
            paths (list): List of image paths.
            labels (list): List of image labels.
            batch_size (int): Batch size.
            model_name (str): CNN model name for preprocessing.
            preprocess_fn (callable, optional): Function for applying preprocessing.
            augment_fn (callable, optional): Function for applying data augmentation.
            shuffle (bool): Whether to shuffle the data.
        """
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.model_name = model_name
        self.preprocess_fn = preprocess_fn
        self.augment_fn = augment_fn
        self.shuffle = shuffle
        self.n = len(paths)
        self.indices = np.arange(self.n)

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        """Create a new iterator."""
        return self

    def __next__(self):
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
        batch_indices = self.indices[self._current_idx:min(self._current_idx + self.batch_size, self.n)]
        self._current_idx += self.batch_size

        # Load and process batch
        batch_images = []
        batch_labels = []

        for i in batch_indices:
            image = load_image(self.paths[i])
            if image is None:
                continue

            # Apply preprocessing if available
            if self.preprocess_fn:
                image = self.preprocess_fn(image)

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
            generator: A generator yielding (batch_images, batch_labels).
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


def load_and_preprocess_dataset(paths, labels, model_name, preprocess_fn=None, augment_fn=None):
    """
    Load and preprocess a full dataset into memory.

    Args:
        paths (list): List of image paths.
        labels (list): List of image labels.
        model_name (str): CNN model name for preprocessing.
        preprocess_fn (callable, optional): Function for applying preprocessing.
        augment_fn (callable, optional): Function for applying data augmentation.

    Returns:
        tuple: (preprocessed_images, one_hot_labels)
    """
    images = []
    for path in paths:
        image = load_image(path)
        if image is None:
            continue

        image = resize_image(image, IMG_SIZE[:2])

        # Apply preprocessing if available
        if preprocess_fn:
            image = preprocess_fn(image)

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