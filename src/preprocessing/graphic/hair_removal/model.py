import tensorflow as tf
from tensorflow.keras import layers as L


class SEResBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation Residual Block"""

    def __init__(self, filters, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

        # Main path
        self.conv1 = L.Conv2D(filters, 3, padding="same")
        self.bn1 = L.BatchNormalization()
        self.conv2 = L.Conv2D(filters, 3, padding="same")
        self.bn2 = L.BatchNormalization()
        self.relu = L.ReLU()

        # Squeeze-and-Excitation
        self.gap = L.GlobalAveragePooling2D()
        self.d1 = L.Dense(filters // ratio, activation="relu")
        self.d2 = L.Dense(filters, activation="sigmoid")

        # Projection shortcut
        self.proj = None
        self.bn_proj = None

    def build(self, input_shape):
        in_ch = input_shape[-1]
        if in_ch != self.filters:
            self.proj = L.Conv2D(self.filters, 1, padding="same")
            self.bn_proj = L.BatchNormalization()

    def call(self, inputs, training=None):
        # Shortcut path
        shortcut = inputs
        if self.proj is not None:
            shortcut = self.proj(shortcut)
            shortcut = self.bn_proj(shortcut, training=training)

        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Squeeze-and-Excitation
        se = self.gap(x)
        se = self.d1(se)
        se = self.d2(se)
        se = tf.expand_dims(tf.expand_dims(se, 1), 1)

        # Apply SE weights
        x = x * se

        # Residual connection
        x = x + shortcut
        x = self.relu(x)

        return x


def create_chimeranet(img_size=448, num_classes=1):
    """
    Create ChimeraNet model as described in the paper:
    "Hair removal in dermoscopy images using variational autoencoders"

    Architecture:
    - EfficientNetB5 encoder with skip connections
    - Decoder with transposed convolutions and SE-ResBlocks
    - Skip connections at multiple resolutions
    """

    inputs = L.Input(shape=(img_size, img_size, 3))

    # Create EfficientNetB5 backbone
    backbone = tf.keras.applications.EfficientNetB5(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet"
    )

    # Get intermediate outputs for skip connections
    # For 448x448 input, EfficientNetB5 produces these resolutions:
    skip_layers = [
        ("block2b_expand_activation", 112),  # 112×112
        ("block3b_expand_activation", 56),  # 56×56
        ("block4e_expand_activation", 28),  # 28×28
        ("block6f_expand_activation", 14),  # 14×14
    ]

    # Build model with multiple outputs
    skip_outputs = []
    for layer_name, _ in skip_layers:
        skip_outputs.append(backbone.get_layer(layer_name).output)
    skip_outputs.append(backbone.output)  # Final output at 14×14

    # Create model that outputs all needed features
    encoder = tf.keras.Model(inputs=backbone.input, outputs=skip_outputs)

    # Get all encoder outputs
    encoder_outputs = encoder(inputs)
    skip_112 = encoder_outputs[0]  # 112×112
    skip_56 = encoder_outputs[1]  # 56×56
    skip_28 = encoder_outputs[2]  # 28×28
    skip_14 = encoder_outputs[3]  # 14×14
    x = encoder_outputs[4]  # Final backbone output 14×14

    # Decoder path
    # Stage 1: 14×14 → 28×28
    x = L.Conv2DTranspose(512, kernel_size=2, strides=2, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # Concatenate with skip from block4e (28×28)
    x = L.Concatenate()([x, skip_28])
    x = L.Conv2D(512, 1, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # SE-ResBlock
    se1 = SEResBlock(512)
    x = se1(x)

    # Stage 2: 28×28 → 56×56
    x = L.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # Concatenate with skip from block3b (56×56)
    x = L.Concatenate()([x, skip_56])
    x = L.Conv2D(256, 1, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # SE-ResBlock
    se2 = SEResBlock(256)
    x = se2(x)

    # Stage 3: 56×56 → 112×112
    x = L.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # Concatenate with skip from block2b (112×112)
    x = L.Concatenate()([x, skip_112])
    x = L.Conv2D(128, 1, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # SE-ResBlock
    se3 = SEResBlock(128)
    x = se3(x)

    # Stage 4: 112×112 → 224×224
    x = L.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # SE-ResBlock
    se4 = SEResBlock(64)
    x = se4(x)

    # Stage 5: 224×224 → 448×448
    x = L.Conv2DTranspose(32, kernel_size=2, strides=2, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # Concatenate with original input
    x = L.Concatenate()([x, inputs])

    # Final SE-ResBlock
    se5 = SEResBlock(32)
    x = se5(x)

    # Final layers
    x = L.Conv2D(32, 3, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    x = L.Dropout(0.5)(x)

    # Output layer
    if num_classes == 1:
        outputs = L.Conv2D(1, 1, activation='sigmoid', name='hair_mask')(x)
    else:
        outputs = L.Conv2D(num_classes, 1, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ChimeraNet')
    return model


class ChimeraNet(tf.keras.Model):
    """Subclassed version of ChimeraNet for compatibility"""

    def __init__(self, img_size=448, num_classes=1):
        super().__init__()
        self.model = create_chimeranet(img_size, num_classes)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def build(self, input_shape):
        super().build(input_shape)
        # Build the internal model
        dummy_input = tf.zeros((1,) + input_shape[1:])
        _ = self.model(dummy_input)