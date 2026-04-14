"""
CNN model definitions and loading utilities.
"""

import os

import tensorflow as tf
from tensorflow.keras.applications import VGG19, InceptionV3, ResNet50, Xception, EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from config import cfg, FINE_TUNING_AT_LAYER


def focal_loss(gamma=2.0, alpha=None):
    """
    Focal Loss for multi-class classification (Lin et al., 2017 – RetinaNet).

    Down-weights easy (high-confidence) examples and focuses learning on hard
    (low-confidence) examples. Effective for HAM10000's severe class imbalance
    (~67% nv class). Use via cfg.use_focal_loss=True in config.py.

    Args:
        gamma: Focusing parameter (typical range 0.5–5.0; 2.0 is the default).
        alpha: Optional per-class weight tensor of shape (cfg.num_classes,).
               Pass class-balanced weights to further address imbalance.

    Returns:
        Keras loss function.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        if alpha is not None:
            weight = weight * alpha
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn


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
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
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
    return "_".join(components)


def load_or_create_cnn(model_name, mode='classifier', fine_tune=True,
                       weights='imagenet', save_path=None):
    """
    Load an existing CNN model or create a new one.

    Args:
        model_name (str): Name of the CNN model ('VGG19', 'Inception', 'ResNet', 'Xception', 'EfficientNet').
        mode (str): 'classifier' or 'extractor'.
        fine_tune (bool): Whether to fine-tune the model.
        weights (str): Pre-trained weights to use.
        save_path (str, optional): Path to save/load the model.

    Returns:
        tuple: (model, loaded) where loaded is True if model was loaded from disk.
    """
    if save_path and os.path.exists(save_path):
        print(f"Loading existing model from: {save_path}")
        return load_model(save_path), True

    print(f"Creating new {model_name} model as {mode}...")

    img_size = cfg.img_size
    fine_tune_at = FINE_TUNING_AT_LAYER.get(model_name, 0)

    if model_name == "VGG19":
        base_model = VGG19(weights=weights, include_top=False, input_shape=img_size)
    elif model_name == "Inception":
        base_model = InceptionV3(weights=weights, include_top=False, input_shape=img_size)
    elif model_name == "ResNet":
        base_model = ResNet50(weights=weights, include_top=False, input_shape=img_size)
    elif model_name == "Xception":
        base_model = Xception(weights=weights, include_top=False, input_shape=img_size)
    elif model_name == "EfficientNet":
        base_model = EfficientNetB4(weights=weights, include_top=False, input_shape=img_size)
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from 'VGG19', 'Inception', 'ResNet', 'Xception', 'EfficientNet'"
        )

    if fine_tune:
        print(f"Applying fine-tuning starting from layer {fine_tune_at}")
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        base_model.trainable = False

    if mode == 'classifier':
        x = GlobalAveragePooling2D(name='gap')(base_model.output)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout')(x)
        predictions = Dense(cfg.num_classes, activation='softmax', name='predictions')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        optimizer = Adam(learning_rate=0.0001)
        if cfg.use_focal_loss:
            loss_fn = focal_loss(gamma=2.0)
        elif cfg.label_smoothing > 0:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=cfg.label_smoothing
            )
        else:
            loss_fn = 'categorical_crossentropy'

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        )
    else:
        model = base_model

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
    extraction_layer = {
        "VGG19":        "block5_pool",
        "Inception":    "mixed10",
        "ResNet":       "avg_pool",
        "Xception":     "avg_pool",
        "EfficientNet": "top_activation"
    }.get(model_name)

    if not extraction_layer:
        raise ValueError(f"Unsupported model: {model_name}")

    selected_output = model.get_layer(extraction_layer).output
    features = GlobalAveragePooling2D()(selected_output)
    return Model(
        inputs=model.input,
        outputs=features,
        name=f"{model_name.lower()}_feature_extractor"
    )


def get_feature_extractor_model(model_name, fine_tune=True, weights='imagenet', save_path=None):
    """
    Get a feature extractor model based on a pre-trained CNN.

    Args:
        model_name (str): Name of the CNN model.
        fine_tune (bool): Whether to fine-tune the model.
        weights (str): Pre-trained weights to use.
        save_path (str, optional): Path to save/load the model.

    Returns:
        tuple: (feature_extractor, loaded)
    """
    base_model, loaded = load_or_create_cnn(
        model_name=model_name,
        mode='extractor',
        fine_tune=fine_tune,
        weights=weights,
        save_path=save_path
    )

    if loaded:
        return base_model, True

    feature_extractor = create_feature_extractor(base_model, model_name)

    if save_path and not os.path.exists(save_path):
        print(f"Saving feature extractor to: {save_path}")
        feature_extractor.save(save_path)

    return feature_extractor, False


def find_trained_cnn_model(results_dir='./results'):
    """
    Find the best trained CNN model based on macro-F1 in model_performance_summary.csv.

    Args:
        results_dir (str): Base results directory.

    Returns:
        str: Path to the best model file, or None if not found.
    """
    import pandas as pd
    from utils.result_naming import cnn_result_dir

    base_dir = cnn_result_dir(base_dir=results_dir)
    final_models_dir = os.path.join(base_dir, "final_models")
    summary_csv = os.path.join(final_models_dir, "model_performance_summary.csv")

    if not os.path.isfile(summary_csv):
        print(f"Performance summary not found: {summary_csv}")
        return None

    df = pd.read_csv(summary_csv)

    if 'model_idx' not in df.columns or 'macro_avg_f1' not in df.columns:
        print("Required columns missing from performance summary CSV.")
        return None

    best_row = df.sort_values(by='macro_avg_f1', ascending=False).iloc[0]
    best_model_idx = int(best_row['model_idx'])

    best_model_path = os.path.join(
        final_models_dir, f"model_{best_model_idx}", "final_cnn_model.keras"
    )

    if os.path.exists(best_model_path):
        print(f"Best CNN model found: {best_model_path}")
        return best_model_path

    print(f"Model listed in summary but file not found: {best_model_path}")
    return None


def get_feature_extractor_from_cnn(feature_extractor_save_path, cnn_model_path=None):
    """
    Obtain a feature extractor from a trained CNN, or create one from ImageNet weights.

    Args:
        feature_extractor_save_path (str): Path to save/load the feature extractor.
        cnn_model_path (str, optional): Path to a trained CNN model; auto-detected if None.

    Returns:
        tuple: (feature_extractor, from_pretrained) where from_pretrained is True
               when the extractor was derived from a trained CNN.
    """
    if os.path.exists(feature_extractor_save_path):
        print(f"Loading existing feature extractor: {feature_extractor_save_path}")
        return tf.keras.models.load_model(feature_extractor_save_path), True

    if cnn_model_path is None:
        cnn_model_path = find_trained_cnn_model()

    if cnn_model_path and os.path.exists(cnn_model_path):
        print(f"Converting trained CNN to feature extractor: {cnn_model_path}")
        try:
            trained_model = tf.keras.models.load_model(cnn_model_path)

            # Find the last non-Dense layer
            extraction_layer = None
            dense_found = False
            for i in range(len(trained_model.layers) - 1, -1, -1):
                layer = trained_model.layers[i]
                if 'dense' in layer.name.lower():
                    dense_found = True
                elif dense_found:
                    extraction_layer = layer
                    break

            if extraction_layer is None:
                extraction_layer = trained_model.layers[-2]

            print(f"Extracting features from layer '{extraction_layer.name}'")

            feature_extractor = tf.keras.Model(
                inputs=trained_model.input,
                outputs=extraction_layer.output
            )

            os.makedirs(os.path.dirname(feature_extractor_save_path), exist_ok=True)
            feature_extractor.save(feature_extractor_save_path)
            print(f"Feature extractor saved: {feature_extractor_save_path}")

            return feature_extractor, True

        except Exception as exc:
            print(f"Failed to convert CNN to feature extractor: {exc}")
            print("Falling back to ImageNet weights...")

    print("Creating feature extractor from ImageNet weights...")
    feature_extractor, _ = get_feature_extractor_model(
        model_name=cfg.cnn_model,
        fine_tune=cfg.use_fine_tuning,
        save_path=feature_extractor_save_path
    )

    return feature_extractor, False


# ---------------------------------------------------------------------------
# Grad-CAM – Gradient-weighted Class Activation Mapping
# Selvaraju et al. (2017, ICCV). Produces spatial heatmaps that show which
# image regions drove a specific class prediction.
# ---------------------------------------------------------------------------

GRADCAM_LAYER = {
    "VGG19":        "block5_conv4",
    "Inception":    "mixed10",
    "ResNet":       "conv5_block3_out",
    "Xception":     "block14_sepconv2_act",
    "EfficientNet": "top_activation",
}


def compute_gradcam(model, img_array, class_idx, model_name):
    """
    Compute a Grad-CAM heatmap for the given image and class.

    Args:
        model: Trained Keras classifier model.
        img_array: Input image as float32 array of shape (1, H, W, 3) in [0,1].
        class_idx: Integer index of the target class.
        model_name: One of 'VGG19', 'Inception', 'ResNet', 'Xception', 'EfficientNet'.

    Returns:
        2-D numpy array in [0, 1] — the normalized CAM heatmap.
    """
    layer_name = GRADCAM_LAYER.get(model_name)
    if layer_name is None:
        raise ValueError(f"No Grad-CAM layer configured for model '{model_name}'")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    cam = tf.nn.relu(cam)
    cam_max = tf.reduce_max(cam)
    return (cam / (cam_max + 1e-8)).numpy()


def save_gradcam_visualizations(model, model_name, image_paths, y_true, y_pred,
                                y_pred_prob, class_names, result_dir, n_per_class=3):
    """
    Save Grad-CAM heatmap overlays for the highest-confidence correct and incorrect
    predictions, one set per class.

    Args:
        model: Trained Keras classifier.
        model_name: Backbone name (e.g. 'ResNet').
        image_paths: Array of test image paths.
        y_true: True class indices (1-D int array).
        y_pred: Predicted class indices (1-D int array).
        y_pred_prob: Softmax probabilities (2-D float array, n_samples × n_classes).
        class_names: List of class name strings.
        result_dir: Directory where gradcam/ sub-folder will be created.
        n_per_class: How many examples to save per class (default: 3).
    """
    import cv2
    import numpy as np

    gradcam_dir = os.path.join(result_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    h, w = cfg.img_size[:2]
    n_classes = len(class_names) if class_names else int(y_pred_prob.shape[1])

    for class_idx in range(n_classes):
        class_label = class_names[class_idx] if class_names else str(class_idx)

        correct_mask = (y_true == class_idx) & (y_pred == class_idx)
        wrong_mask   = (y_true == class_idx) & (y_pred != class_idx)

        for subset_name, mask in [("correct", correct_mask), ("wrong", wrong_mask)]:
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            conf = y_pred_prob[indices, y_pred[indices]]
            top_indices = indices[np.argsort(conf)[::-1][:n_per_class]]

            for rank, idx in enumerate(top_indices):
                path = image_paths[idx]
                img_bgr = cv2.imread(str(path))
                if img_bgr is None:
                    continue
                img_bgr = cv2.resize(img_bgr, (w, h))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_array = img_rgb.astype(np.float32) / 255.0
                img_input = np.expand_dims(img_array, axis=0)

                try:
                    cam = compute_gradcam(model, img_input, class_idx, model_name)
                    cam_resized = cv2.resize(cam, (w, h))
                    heatmap = cv2.applyColorMap(
                        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
                    )
                    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
                    fname = f"{class_label}_{subset_name}_rank{rank + 1}.jpg"
                    cv2.imwrite(os.path.join(gradcam_dir, fname), overlay)
                except Exception as exc:
                    print(f"Grad-CAM failed for {path}: {exc}")

    print(f"Grad-CAM overlays saved to: {gradcam_dir}")
