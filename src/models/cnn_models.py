"""
CNN model definitions and loading utilities.
"""

import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    InceptionV3, Xception, ConvNeXtTiny, EfficientNetV2S,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.models import Model, load_model

from config import cfg, FINE_TUNING_AT_LAYER


# AdamW resolution:
# 1) tfa.optimizers.AdamW — classic optimizer API, works with mixed precision
#    under TF 2.10 (the experimental optimizer has an AutoCastVariable tracing bug).
# 2) tf.keras.optimizers.AdamW — TF >= 2.11 native.
# 3) tf.keras.optimizers.experimental.AdamW — last-resort TF 2.10 fallback.
def _resolve_adamw():
    try:
        import tensorflow_addons as tfa
        return tfa.optimizers.AdamW
    except Exception:
        pass
    native = getattr(tf.keras.optimizers, 'AdamW', None)
    if native is not None:
        return native
    exp = getattr(tf.keras.optimizers.experimental, 'AdamW', None)
    if exp is not None:
        return exp
    raise ImportError("No AdamW optimizer available — install tensorflow_addons or TF>=2.11.")


_AdamW_cls = _resolve_adamw()


def focal_loss(gamma=2.0, alpha=None):
    """
    Focal Loss for multi-class classification (Lin et al., 2017 – RetinaNet).
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        if alpha is not None:
            weight = weight * alpha
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn


def resolve_loss():
    """Pick the loss function from the active cfg flags."""
    if cfg.use_focal_loss:
        return focal_loss(gamma=2.0)
    if cfg.label_smoothing > 0:
        return tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.label_smoothing
        )
    return 'categorical_crossentropy'


def get_callbacks(save_path, tensorboard_log_dir=None):
    """EarlyStopping + ModelCheckpoint (+ optional TensorBoard)."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
    ]
    if tensorboard_log_dir:
        callbacks.append(
            TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch',
                profile_batch=0,
            )
        )
    return callbacks


def create_model_name(base_model_name, mode, use_fine_tuning, use_preprocessing):
    components = [
        base_model_name.lower(),
        mode,
        f"ft_{use_fine_tuning}",
        f"preproc_{use_preprocessing}",
    ]
    return "_".join(components)


_BACKBONE_CTORS = {
    "Inception":       InceptionV3,
    "Xception":        Xception,
    "ConvNeXt":        ConvNeXtTiny,
    "EfficientNetV2S": EfficientNetV2S,
}


def _build_base_model(model_name, weights, img_size):
    ctor = _BACKBONE_CTORS.get(model_name)
    if ctor is None:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from {list(_BACKBONE_CTORS)}"
        )
    # Keras 2.10's ConvNeXt LayerScale layer creates float32 weights that
    # multiply float16 inputs under mixed_float16, raising a TypeError at build
    # time. Temporarily force float32 policy while constructing ConvNeXt, then
    # restore the original policy so the rest of the graph keeps using fp16.
    if model_name == "ConvNeXt":
        prev = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy('float32')
            base = ctor(weights=weights, include_top=False, input_shape=img_size)
        finally:
            tf.keras.mixed_precision.set_global_policy(prev)
        return base
    return ctor(weights=weights, include_top=False, input_shape=img_size)


def _build_classifier_head(base_model, num_classes):
    """
    Modern BN-regularized head with adaptive width (§1.4 of A2 audit).
    GAP → BN → Dense(feat_dim) → BN → Dropout
        → Dense(256) → BN → Dropout
        → Dense(num_classes, softmax, float32)
    """
    feat_dim = int(base_model.output_shape[-1])
    x = GlobalAveragePooling2D(name='gap')(base_model.output)
    x = BatchNormalization(name='bn_gap')(x)
    x = Dense(feat_dim, activation='relu', name='fc1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    x = Dropout(0.3, name='dropout1')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = BatchNormalization(name='bn_fc2')(x)
    x = Dropout(0.3, name='dropout2')(x)
    predictions = Dense(
        num_classes, activation='softmax',
        dtype='float32', name='predictions',
    )(x)
    return predictions


def _make_adamw(initial_lr, decay_steps, weight_decay):
    """AdamW with CosineDecay schedule."""
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=max(1, decay_steps),
        alpha=1e-6,
    )
    return _AdamW_cls(
        learning_rate=schedule,
        weight_decay=weight_decay,
    )


def compile_phase1(model, weight_decay=None):
    """
    Phase 1: head-only training. Backbone frozen by caller.
    AdamW at lr=1e-3, cosine over cfg.warmup_epochs worth of steps is done
    implicitly via a separate .fit() call; here we use a constant-ish LR via
    CosineDecay over the warmup span (caller passes decay_steps).
    """
    wd = cfg.weight_decay if weight_decay is None else weight_decay
    opt = _AdamW_cls(learning_rate=1e-3, weight_decay=wd)
    model.compile(
        optimizer=opt,
        loss=resolve_loss(),
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
    )


def compile_phase2(model, n_train, batch_size, num_epochs, weight_decay=None):
    """
    Phase 2: unfrozen backbone + head. AdamW with CosineDecay over the
    remaining training span.
    """
    wd = cfg.weight_decay if weight_decay is None else weight_decay
    steps_per_epoch = max(1, math.ceil(n_train / max(1, batch_size)))
    decay_steps = steps_per_epoch * max(1, num_epochs)
    opt = _make_adamw(initial_lr=1e-4, decay_steps=decay_steps, weight_decay=wd)
    model.compile(
        optimizer=opt,
        loss=resolve_loss(),
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
    )


def load_or_create_cnn(model_name, mode='classifier', fine_tune=True,
                       weights='imagenet', save_path=None, n_train=None):
    """
    Load an existing CNN model or create a new one.

    When mode='classifier', the returned model is compiled for Phase 1
    (head-only, backbone frozen) — progressive unfreezing is driven by
    `train.py` calling `compile_phase2` after the warmup epochs.

    Args:
        n_train: Training-set size used to compute Phase-2 decay_steps. Only
            needed when mode='classifier' and use_fine_tuning=True at the
            caller's discretion; not used in Phase-1 compilation here.
    """
    if save_path and os.path.exists(save_path):
        print(f"Loading existing model from: {save_path}")
        return load_model(save_path), True

    print(f"Creating new {model_name} model as {mode}...")

    img_size = cfg.img_size
    base_model = _build_base_model(model_name, weights, img_size)

    if mode == 'classifier':
        # Phase 1: freeze entire backbone.
        base_model.trainable = False
        predictions = _build_classifier_head(base_model, cfg.num_classes)
        model = Model(inputs=base_model.input, outputs=predictions)
        compile_phase1(model)
    else:
        model = base_model

    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    return model, False


def unfreeze_from(model, model_name):
    """
    Unfreeze backbone layers from FINE_TUNING_AT_LAYER[model_name] onward.
    BatchNormalization layers stay frozen to avoid destabilising imagenet
    statistics during fine-tuning.
    """
    at = FINE_TUNING_AT_LAYER.get(model_name, 0)
    # The classifier has the head appended; identify the backbone layers by
    # walking until we hit the head's first layer (named 'gap').
    backbone_layers = []
    for layer in model.layers:
        if layer.name == 'gap':
            break
        backbone_layers.append(layer)

    for i, layer in enumerate(backbone_layers):
        if i < at:
            layer.trainable = False
        else:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    # Head always trainable.
    for layer in model.layers[len(backbone_layers):]:
        layer.trainable = True


def create_feature_extractor(model, model_name):
    features = GlobalAveragePooling2D()(model.output)
    return Model(
        inputs=model.input,
        outputs=features,
        name=f"{model_name.lower()}_feature_extractor",
    )


def get_feature_extractor_model(model_name, fine_tune=True, weights='imagenet', save_path=None):
    base_model, loaded = load_or_create_cnn(
        model_name=model_name,
        mode='extractor',
        fine_tune=fine_tune,
        weights=weights,
        save_path=save_path,
    )

    if loaded:
        return base_model, True

    feature_extractor = create_feature_extractor(base_model, model_name)

    if save_path and not os.path.exists(save_path):
        print(f"Saving feature extractor to: {save_path}")
        feature_extractor.save(save_path)

    return feature_extractor, False


def find_trained_cnn_model(results_dir='./results'):
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
    if os.path.exists(feature_extractor_save_path):
        print(f"Loading existing feature extractor: {feature_extractor_save_path}")
        return tf.keras.models.load_model(feature_extractor_save_path), True

    if cnn_model_path is None:
        cnn_model_path = find_trained_cnn_model()

    if cnn_model_path and os.path.exists(cnn_model_path):
        print(f"Converting trained CNN to feature extractor: {cnn_model_path}")
        try:
            trained_model = tf.keras.models.load_model(cnn_model_path)

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
                outputs=extraction_layer.output,
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
        save_path=feature_extractor_save_path,
    )

    return feature_extractor, False


# ---------------------------------------------------------------------------
# Grad-CAM++ and Score-CAM (replace standard Grad-CAM per A2 audit §1.9/§2.7)
# Chattopadhay et al. 2018 (Grad-CAM++); Wang et al. 2020 (Score-CAM).
# ---------------------------------------------------------------------------

GRADCAM_LAYER = {
    "Inception":       "mixed10",
    "Xception":        "block14_sepconv2_act",
    "ConvNeXt":        None,   # auto-detect last 4D output
    "EfficientNetV2S": "top_activation",
}


def _resolve_cam_layer(model, model_name):
    """
    Return a Keras layer suitable for CAM extraction for the active model.

    If GRADCAM_LAYER has an explicit name, use it. Otherwise scan from the
    end of the model for the last layer whose output is a 4-D spatial tensor.
    """
    name = GRADCAM_LAYER.get(model_name)
    if name is not None:
        try:
            return model.get_layer(name)
        except ValueError:
            pass

    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            return layer
    raise ValueError(f"No 4-D output layer found for model '{model_name}'")


def compute_gradcam_pp(model, img_array, class_idx, model_name):
    """
    Grad-CAM++ (Chattopadhay et al. 2018).

    Higher-order gradient weighting produces tighter, better-localized
    heatmaps than standard Grad-CAM for multiple or small target objects.
    """
    layer = _resolve_cam_layer(model, model_name)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output, model.output],
    )

    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_outputs, predictions = grad_model(img_tensor)
                y_c = predictions[:, class_idx]
            grads_1 = tape1.gradient(y_c, conv_outputs)
        grads_2 = tape2.gradient(grads_1, conv_outputs)
    grads_3 = tape3.gradient(grads_2, conv_outputs)

    # Per-channel weights (Chattopadhay eq. 19).
    sum_feature = tf.reduce_sum(conv_outputs[0], axis=(0, 1))
    g2 = grads_2[0]
    g3 = grads_3[0]
    denom = 2.0 * g2 + sum_feature * g3
    denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
    alphas = g2 / denom

    weights = tf.reduce_sum(alphas * tf.nn.relu(grads_1[0]), axis=(0, 1))
    cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    cam = tf.nn.relu(cam)
    cam_max = tf.reduce_max(cam)
    return tf.cast(cam / (cam_max + 1e-8), tf.float32).numpy()


def compute_score_cam(model, img_array, class_idx, model_name, max_channels=64):
    """
    Score-CAM (Wang et al. 2020).

    Gradient-free: weights each feature-map channel by the class score of
    the image masked with that channel's upsampled activation.

    max_channels: since Score-CAM needs one forward pass per channel, cap
    at `max_channels` top-activated channels to keep runtime manageable.
    """
    layer = _resolve_cam_layer(model, model_name)

    feature_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    img_tensor = tf.cast(img_array, tf.float32)
    feature_maps = feature_model(img_tensor)[0].numpy()   # H' × W' × C
    h, w, c = feature_maps.shape

    activations = feature_maps.sum(axis=(0, 1))
    top_ch = np.argsort(activations)[::-1][:min(max_channels, c)]

    img_np = img_array[0]
    H, W = img_np.shape[:2]

    masked_batch = np.zeros((len(top_ch), H, W, img_np.shape[2]), dtype=np.float32)
    for i, ch in enumerate(top_ch):
        fmap = feature_maps[..., ch]
        fmap_max = fmap.max()
        fmap_min = fmap.min()
        if fmap_max - fmap_min < 1e-8:
            continue
        norm = ((fmap - fmap_min) / (fmap_max - fmap_min)).astype(np.float32)
        import cv2
        upsampled = cv2.resize(norm, (W, H))
        masked_batch[i] = img_np * upsampled[..., None]

    scores = model.predict(masked_batch, verbose=0)[:, class_idx]

    cam = np.zeros((h, w), dtype=np.float32)
    for i, ch in enumerate(top_ch):
        cam += scores[i] * feature_maps[..., ch]
    cam = np.maximum(cam, 0.0)
    cam_max = cam.max()
    return cam / (cam_max + 1e-8)


def save_gradcam_visualizations(model, model_name, image_paths, y_true, y_pred,
                                y_pred_prob, class_names, result_dir, n_per_class=3):
    """
    Save side-by-side visualizations (original | Grad-CAM++ | Score-CAM) for
    the highest-confidence correct and incorrect predictions per class.
    """
    import cv2

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
                    cam_pp = compute_gradcam_pp(model, img_input, class_idx, model_name)
                    cam_sc = compute_score_cam(model, img_input, class_idx, model_name)

                    def _overlay(cam):
                        cam_r = cv2.resize(np.nan_to_num(cam).astype(np.float32), (w, h))
                        hm = cv2.applyColorMap(
                            np.uint8(255 * cam_r), cv2.COLORMAP_JET
                        )
                        return cv2.addWeighted(img_bgr, 0.6, hm, 0.4, 0)

                    overlay_pp = _overlay(cam_pp)
                    overlay_sc = _overlay(cam_sc)
                    grid = np.concatenate([img_bgr, overlay_pp, overlay_sc], axis=1)
                    fname = f"{class_label}_{subset_name}_rank{rank + 1}.jpg"
                    cv2.imwrite(os.path.join(gradcam_dir, fname), grid)
                except Exception as exc:
                    print(f"CAM failed for {path}: {exc}")

    print(f"CAM overlays (Grad-CAM++ + Score-CAM) saved to: {gradcam_dir}")
