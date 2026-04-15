"""GPU memory configuration utilities shared by both pipeline modules."""

import tensorflow as tf


def setup_gpu_memory():
    """Enable TensorFlow memory growth and mixed-precision on GPU."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth must be set before GPUs have been initialized.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")

            # Mixed-precision: uses float16 for compute and float32 for
            # variables — typically 1.5-2× faster on Volta/Turing/Ampere GPUs.
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed-precision (float16) enabled.")
            except Exception as mp_err:
                print(f"Mixed-precision not available: {mp_err}")

        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")
