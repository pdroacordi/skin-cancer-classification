"""GPU memory configuration utilities shared by both pipeline modules."""

import tensorflow as tf


def setup_gpu_memory():
    """Enable TensorFlow memory growth to avoid OOM errors on GPU."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth must be set before GPUs have been initialized.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")
