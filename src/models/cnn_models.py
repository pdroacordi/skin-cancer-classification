"""
CNN model definitions and loading utilities.
"""

import datetime
import os
import sys

import tensorflow as tf
from tensorflow.keras.applications import VGG19, InceptionV3, ResNet50, Xception
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from skincancer.src.config import USE_DATA_AUGMENTATION

sys.path.append('..')
from config import (
    NUM_CLASSES,
    IMG_SIZE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    FINE_TUNING_AT_LAYER
)


def get_callbacks(save_path, tensorboard_log_dir=None):
    """
    Get a list of callbacks for model training.

    Args:
        save_path (str): Path to save the best model.
        tensorboard_log_dir (str, optional): Directory for TensorBoard logs.

    Returns:
        list: List of Keras callbacks.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Add TensorBoard callback if log directory is provided
    if tensorboard_log_dir:
        callbacks.append(
            TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch',
                profile_batch=0
            )
        )

    return callbacks


def create_model_name(base_model_name, mode, use_fine_tuning, use_preprocessing):
    """
    Create a descriptive model name based on configuration.

    Args:
        base_model_name (str): Base CNN model name.
        mode (str): 'classifier' or 'extractor'.
        use_fine_tuning (bool): Whether fine-tuning is used.
        use_preprocessing (bool): Whether preprocessing is used.

    Returns:
        str: Model name.
    """
    components = [
        base_model_name.lower(),
        mode,
        f"ft_{use_fine_tuning}",
        f"preproc_{use_preprocessing}"
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    return f"{'_'.join(components)}_{timestamp}"


def load_or_create_cnn(model_name, mode='classifier', fine_tune=True,
                       weights='imagenet', save_path=None):
    """
    Load an existing CNN model or create a new one.

    Args:
        model_name (str): Name of the CNN model ('VGG19', 'Inception', 'ResNet', 'Xception').
        mode (str): 'classifier' or 'extractor'.
        fine_tune (bool): Whether to fine-tune the model.
        weights (str): Pre-trained weights to use.
        save_path (str, optional): Path to save/load the model.

    Returns:
        tuple: (model, loaded) where loaded is True if model was loaded from disk.
    """
    # Try to load the model if save_path is provided and exists
    if save_path and os.path.exists(save_path):
        print(f"Loading existing model from: {save_path}")
        return load_model(save_path), True

    # Create a new model
    print(f"Creating new {model_name} model as {mode}...")

    # Create the base model
    if model_name == "VGG19":
        base_model = VGG19(weights=weights, include_top=False, input_shape=IMG_SIZE)
        fine_tune_at = FINE_TUNING_AT_LAYER["VGG19"]
    elif model_name == "Inception":
        base_model = InceptionV3(weights=weights, include_top=False, input_shape=IMG_SIZE)
        fine_tune_at = FINE_TUNING_AT_LAYER["Inception"]
    elif model_name == "ResNet":
        base_model = ResNet50(weights=weights, include_top=False, input_shape=IMG_SIZE)
        fine_tune_at = FINE_TUNING_AT_LAYER["ResNet"]
    elif model_name == "Xception":
        base_model = Xception(weights=weights, include_top=False, input_shape=IMG_SIZE)
        fine_tune_at = FINE_TUNING_AT_LAYER["Xception"]
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from 'VGG19', 'Inception', 'ResNet', or 'Xception'")

    # Apply fine-tuning strategy
    if fine_tune:
        print(f"Applying fine-tuning starting from layer {fine_tune_at}")
        # Freeze earlier layers, unfreeze later layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        # Freeze all layers in the base model
        base_model.trainable = False

    # Add classification head for end-to-end classifier
    if mode == 'classifier':
        x = GlobalAveragePooling2D(name='gap')(base_model.output)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        optimizer = Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        )
    else:  # Feature extractor mode
        model = base_model

    # Create directory if it doesn't exist
    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    return model, False


def create_feature_extractor(model, model_name):
    """
    Create a feature extractor from a CNN model.

    Args:
        model (tensorflow.keras.Model): A pre-trained CNN model.
        model_name (str): Name of the CNN model.

    Returns:
        tensorflow.keras.Model: Feature extractor model.
    """
    # Define the layer to extract features from
    extraction_layer = {
        "VGG19": "block5_pool",
        "Inception": "mixed10",
        "ResNet": "avg_pool",
        "Xception": "avg_pool"
    }.get(model_name)

    if not extraction_layer:
        raise ValueError(f"Unsupported model: {model_name}")

    # Get the output from the selected layer
    selected_output = model.get_layer(extraction_layer).output

    # Add global average pooling to get a fixed-size feature vector
    features = GlobalAveragePooling2D()(selected_output)

    # Create a new model
    feature_extractor = Model(inputs=model.input, outputs=features, name=f"{model_name.lower()}_feature_extractor")

    return feature_extractor


def get_feature_extractor_model(model_name, fine_tune=True, weights='imagenet', save_path=None):
    """
    Get a feature extractor model based on a pre-trained CNN.

    Args:
        model_name (str): Name of the CNN model ('VGG19', 'Inception', 'ResNet', 'Xception').
        fine_tune (bool): Whether to fine-tune the model.
        weights (str): Pre-trained weights to use.
        save_path (str, optional): Path to save/load the model.

    Returns:
        tuple: (feature_extractor, loaded)
    """
    # First, try to load or create the base model
    base_model, loaded = load_or_create_cnn(
        model_name=model_name,
        mode='extractor',
        fine_tune=fine_tune,
        weights=weights,
        save_path=save_path
    )

    # If we loaded a feature extractor directly
    if loaded:
        return base_model, True

    # Otherwise, create a feature extractor from the base model
    feature_extractor = create_feature_extractor(base_model, model_name)

    # Save the feature extractor if a path is provided
    if save_path and not os.path.exists(save_path):
        print(f"Saving feature extractor to: {save_path}")
        feature_extractor.save(save_path)

    return feature_extractor, False


def fine_tune_feature_extractor(feature_extractor, X_train, y_train, X_val, y_val,
                                epochs=10, batch_size=32, save_path=None,
                                use_augmentation=USE_DATA_AUGMENTATION):
    """
    Fine-tune a feature extractor using a small classification head.

    Args:
        feature_extractor (tensorflow.keras.Model): Feature extractor model.
        X_train (numpy.array): Training images.
        y_train (numpy.array): Training labels (one-hot encoded).
        X_val (numpy.array): Validation images.
        y_val (numpy.array): Validation labels (one-hot encoded).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        save_path (str, optional): Path to save the fine-tuned model.
        use_augmentation (bool): Whether to use data augmentation.

    Returns:
        tensorflow.keras.Model: Fine-tuned feature extractor.
    """
    # Add a classification head for fine-tuning
    x = feature_extractor.output
    predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
    fine_tuning_model = Model(inputs=feature_extractor.input, outputs=predictions)

    # Compile the model
    optimizer = Adam(learning_rate=0.00001)
    fine_tuning_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create callbacks
    callbacks = get_callbacks(save_path) if save_path else []

    # Train the model
    # Apply augmentation during training if requested
    if use_augmentation:
        from utils.augmentation import AugmentationFactory

        medium_augmentation = AugmentationFactory.get_medium_augmentation()

        train_gen = AugmentationFactory.AugmentedDataGenerator(
            X_train, y_train,
            batch_size=batch_size,
            augmentation=medium_augmentation
        )

        fine_tuning_model.fit(
            train_gen,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        fine_tuning_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    # After fine-tuning, we use the feature extractor part of the model
    # Extract the feature extractor (excluding the classification head)
    fine_tuned_feature_extractor = Model(
        inputs=fine_tuning_model.input,
        outputs=fine_tuning_model.layers[-2].output
    )

    # Save the fine-tuned feature extractor if a path is provided
    if save_path:
        print(f"Saving fine-tuned feature extractor to: {save_path}")
        fine_tuned_feature_extractor.save(save_path)

    return fine_tuned_feature_extractor