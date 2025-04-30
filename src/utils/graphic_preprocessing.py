"""
Enhanced graphic preprocessing module with improved segmentation techniques.
This module integrates with the existing pipeline structure.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU
import tensorflow as tf


class GraphicPreprocessing:
    """
    Enhanced graphic preprocessing techniques for skin lesion images.
    """

    def __init__(self, img_size=(299, 299)):
        self.img_size = img_size

    def apply_preprocessing(self, image, use_hair_removal=True,
                          use_contrast_enhancement=True,
                          use_segmentation=False,
                          visualize=False):
        """
        Apply enhanced preprocessing to a skin lesion image.

        Args:
            image: BGR image input
            use_hair_removal: Whether to apply enhanced hair removal
            use_contrast_enhancement: Whether to enhance contrast
            use_segmentation: Whether to segment the lesion
            visualize: Whether to visualize the intermediate steps

        Returns:
            Preprocessed image
        """
        # Make a copy of the input image
        processed = image.copy()

        # Resize the image to the target size
        if processed.shape[0] != self.img_size[0] or processed.shape[1] != self.img_size[1]:
            processed = cv2.resize(processed, self.img_size)

        # Apply hair removal if requested
        if use_hair_removal:
            processed = self.remove_hair_enhanced(processed)

        # Apply contrast enhancement if requested
        if use_contrast_enhancement:
            processed = self.enhance_contrast_adaptive(processed)

        # Apply segmentation if requested
        if use_segmentation:
            processed = self.apply_segmentation(processed)

        # Visualize the preprocessing steps if requested
        if visualize:
            self._visualize_preprocessing(image, processed)

        return processed

    def remove_hair_enhanced(self, image):
        """
        Enhanced hair removal using a combination of morphological operations,
        thresholding, and inpainting.

        Args:
            image: BGR input image

        Returns:
            Image with hair removed
        """
        # Convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create kernels for line detection at different orientations
        kernel_size = 17  # Adjust based on hair thickness

        # Create a bank of line kernels at different orientations
        kernels = []
        for theta in range(0, 180, 15):  # 12 orientations
            theta_rad = np.deg2rad(theta)
            kernel = self._create_line_kernel(kernel_size, theta_rad)
            kernels.append(kernel)

        # Apply blackhat operation with each kernel and combine results
        hair_mask = np.zeros_like(grayscale)
        for kernel in kernels:
            blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
            hair_mask = np.maximum(hair_mask, blackhat)

        # Apply thresholding to identify hair pixels
        _, thresh = cv2.threshold(hair_mask, 10, 255, cv2.THRESH_BINARY)

        # Dilate to ensure complete hair coverage
        dilated_mask = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

        # Inpaint using the mask
        result = cv2.inpaint(image, dilated_mask.astype(np.uint8), inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        return result

    def _create_line_kernel(self, size, theta):
        """
        Create a line-shaped kernel for a specific orientation.

        Args:
            size: Size of the kernel
            theta: Orientation angle in radians

        Returns:
            Line kernel
        """
        # Create a line kernel with the specified orientation
        kernel = np.zeros((size, size), dtype=np.uint8)
        center = size // 2

        # Calculate line endpoints
        x1 = center + int(np.round((size / 2) * np.cos(theta)))
        y1 = center + int(np.round((size / 2) * np.sin(theta)))
        x2 = center - int(np.round((size / 2) * np.cos(theta)))
        y2 = center - int(np.round((size / 2) * np.sin(theta)))

        # Draw the line
        cv2.line(kernel, (x1, y1), (x2, y2), 1, 1)

        return kernel

    def enhance_contrast_adaptive(self, image):
        """
        Enhance contrast using adaptive histogram equalization and gamma correction.

        Args:
            image: BGR input image

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Apply gamma correction to further enhance contrast
        # Calculate an adaptive gamma value based on image brightness
        mean_brightness = np.mean(l_enhanced) / 255.0
        # If image is dark, use gamma < 1 to brighten; if bright, use gamma > 1 to darken
        gamma = 0.9 if mean_brightness < 0.5 else 1.1
        l_gamma = np.power(l_enhanced / 255.0, gamma) * 255.0
        l_gamma = l_gamma.astype(np.uint8)

        # Merge channels and convert back to BGR
        lab_enhanced = cv2.merge([l_gamma, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def apply_segmentation(self, image):
        """
        Apply improved segmentation using pretrained U-Net.

        Args:
            image: BGR input image

        Returns:
            Segmented image with highlighted lesion boundary
        """
        # Get the segmentation model
        model = self._get_segmentation_model()

        # Normalize the image
        input_image = image.astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        # Predict mask
        predicted_mask = model.predict(input_image)[0, :, :, 0]
        binary_mask = (predicted_mask > 0.5).astype(np.uint8)

        # Post-process the mask
        refined_mask = self._post_process_mask(binary_mask)

        # Highlight the lesion boundary on the original image
        result = image.copy()
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

        return result

    def _get_segmentation_model(self):
        """
        Get the segmentation model (U-Net).
        Load from file if available, otherwise create a new one.

        Returns:
            U-Net model
        """
        model_path = os.path.join("models", "unet_skin_lesion.h5")

        if os.path.exists(model_path):
            # Load the model
            custom_objects = {
                'dice_loss': self._dice_loss,
                'dice_coef': self._dice_coef
            }
            model = load_model(model_path, custom_objects=custom_objects)
        else:
            # Create a new model (will require training)
            model = self._build_unet()

            # Create the models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)

            # Save the model
            model.save(model_path)

        return model

    def _build_unet(self):
        """
        Build the U-Net model for skin lesion segmentation.

        Returns:
            U-Net model
        """
        # Input layer
        inputs = Input((self.img_size[0], self.img_size[1], 3))

        # Encoder path
        # Block 1
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        # Save the shape for later use in skip connections
        shape1 = tf.shape(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        drop1 = Dropout(0.1)(pool1)

        # Block 2
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(drop1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        # Save the shape for later use in skip connections
        shape2 = tf.shape(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        drop2 = Dropout(0.2)(pool2)

        # Block 3
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(drop2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        # Save the shape for later use in skip connections
        shape3 = tf.shape(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        drop3 = Dropout(0.3)(pool3)

        # Block 4
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(drop3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        # Save the shape for later use in skip connections
        shape4 = tf.shape(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        drop4 = Dropout(0.4)(pool4)

        # Bridge
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(drop4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Decoder path with dynamic resizing
        # Block 6
        up6 = Conv2D(512, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(drop5))
        # Resize up6 to match conv4 shape
        up6_resized = tf.image.resize(up6, [shape4[1], shape4[2]])
        merge6 = concatenate([conv4, up6_resized], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)

        # Block 7
        up7 = Conv2D(256, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv6))
        # Resize up7 to match conv3 shape
        up7_resized = tf.image.resize(up7, [shape3[1], shape3[2]])
        merge7 = concatenate([conv3, up7_resized], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        # Block 8
        up8 = Conv2D(128, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv7))
        # Resize up8 to match conv2 shape
        up8_resized = tf.image.resize(up8, [shape2[1], shape2[2]])
        merge8 = concatenate([conv2, up8_resized], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        # Block 9
        up9 = Conv2D(64, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv8))
        # Resize up9 to match conv1 shape
        up9_resized = tf.image.resize(up9, [shape1[1], shape1[2]])
        merge9 = concatenate([conv1, up9_resized], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        # Output layer
        outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile model with dice coefficient loss
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss=self._dice_loss,
                      metrics=[self._dice_coef, 'binary_accuracy', MeanIoU(num_classes=2)])

        return model

    def _dice_coef(self, y_true, y_pred, smooth=1):
        """
        Calculate Dice coefficient.

        Args:
            y_true: Ground truth
            y_pred: Predictions
            smooth: Smoothing factor

        Returns:
            Dice coefficient
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    def _dice_loss(self, y_true, y_pred):
        """
        Calculate Dice loss.

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            Dice loss
        """
        return 1 - self._dice_coef(y_true, y_pred)

    def _post_process_mask(self, binary_mask):
        """
        Post-process the segmentation mask.

        Args:
            binary_mask: Binary mask

        Returns:
            Refined binary mask
        """
        # Fill holes
        filled = self._fill_holes(binary_mask)

        # Remove small objects
        cleaned = self._remove_small_objects(filled)

        # Smooth boundaries
        smoothed = self._smooth_boundaries(cleaned)

        return smoothed

    def _fill_holes(self, binary_mask):
        """
        Fill holes in the binary mask.

        Args:
            binary_mask: Binary mask

        Returns:
            Mask with holes filled
        """
        # Ensure the mask is binary
        binary = binary_mask.copy()
        if binary.max() > 1:
            binary = (binary > 127).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a filled mask
        filled_mask = np.zeros_like(binary)

        # Fill each contour
        for contour in contours:
            cv2.drawContours(filled_mask, [contour], 0, 1, -1)

        return filled_mask

    def _remove_small_objects(self, binary_mask, min_size_ratio=0.01):
        """
        Remove small objects from the binary mask.

        Args:
            binary_mask: Binary mask
            min_size_ratio: Minimum size ratio

        Returns:
            Mask with small objects removed
        """
        # Label connected components
        num_labels, labels = cv2.connectedComponents(binary_mask)

        if num_labels == 1:  # Only background
            return binary_mask

        # Calculate minimum size threshold as a ratio of the image size
        min_size = int(min_size_ratio * binary_mask.size)

        # Get component sizes
        for label in range(1, num_labels):
            component_size = np.sum(labels == label)
            if component_size < min_size:
                # Remove small component
                binary_mask[labels == label] = 0

        return binary_mask

    def _smooth_boundaries(self, binary_mask):
        """
        Smooth the boundaries of the binary mask.

        Args:
            binary_mask: Binary mask

        Returns:
            Mask with smoothed boundaries
        """
        # Apply morphological operations to smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)

        return smoothed

    def _visualize_preprocessing(self, original, processed):
        """
        Visualize the preprocessing steps.

        Args:
            original: Original image
            processed: Processed image
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        plt.title('Processed')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def apply_graphic_preprocessing(image, use_hair_removal=True,
                                use_contrast_enhancement=True,
                                use_segmentation=False,
                                visualize=False):
    """
    Wrapper function for the enhanced preprocessing pipeline.
    Matches the signature of the original apply_graphic_preprocessing function.

    Args:
        image: BGR input image
        use_hair_removal: Whether to apply hair removal
        use_contrast_enhancement: Whether to enhance contrast
        use_segmentation: Whether to segment the lesion
        visualize: Whether to visualize the intermediate steps

    Returns:
        Preprocessed image
    """
    preprocessor = GraphicPreprocessing()
    return preprocessor.apply_preprocessing(
        image,
        use_hair_removal=use_hair_removal,
        use_contrast_enhancement=use_contrast_enhancement,
        use_segmentation=use_segmentation,
        visualize=visualize
    )


# Training function for the segmentation model
def train_segmentation_model(train_data_path, val_data_path=None, epochs=50, batch_size=8):
    """
    Train the segmentation model.

    Args:
        train_data_path: Path to training data directory containing 'images' and 'masks' subdirectories
        val_data_path: Path to validation data directory containing 'images' and 'masks' subdirectories
        epochs: Number of epochs
        batch_size: Batch size

    Returns:
        Training history
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt

    # Initialize the preprocessor to get the model
    preprocessor = GraphicPreprocessing()
    model = preprocessor._get_segmentation_model()

    # Define data generators
    data_gen_args = dict(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    mask_gen_args = dict(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    # Create generators
    seed = 42
    image_generator = image_datagen.flow_from_directory(
        train_data_path,
        classes=['images'],
        class_mode=None,
        color_mode='rgb',
        target_size=preprocessor.img_size,
        batch_size=batch_size,
        seed=seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_data_path,
        classes=['masks'],
        class_mode=None,
        color_mode='grayscale',
        target_size=preprocessor.img_size,
        batch_size=batch_size,
        seed=seed
    )

    # Combine generators
    train_generator = zip(image_generator, mask_generator)

    # Create validation generator if validation data path is provided
    if val_data_path:
        val_image_generator = image_datagen.flow_from_directory(
            val_data_path,
            classes=['images'],
            class_mode=None,
            color_mode='rgb',
            target_size=preprocessor.img_size,
            batch_size=batch_size,
            seed=seed
        )

        val_mask_generator = mask_datagen.flow_from_directory(
            val_data_path,
            classes=['masks'],
            class_mode=None,
            color_mode='grayscale',
            target_size=preprocessor.img_size,
            batch_size=batch_size,
            seed=seed
        )

        val_generator = zip(val_image_generator, val_mask_generator)
        validation_steps = val_image_generator.samples // batch_size
    else:
        val_generator = None
        validation_steps = None

    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join("models", "unet_skin_lesion.h5"),
            monitor='val_dice_coef' if val_data_path else 'dice_coef',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coef' if val_data_path else 'dice_coef',
            patience=10,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coef' if val_data_path else 'dice_coef',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max',
            verbose=1
        )
    ]

    # Train the model
    steps_per_epoch = image_generator.samples // batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot Dice coefficient
    plt.subplot(1, 2, 1)
    plt.plot(history.history['dice_coef'], label='Train')
    if val_data_path:
        plt.plot(history.history['val_dice_coef'], label='Validation')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    if val_data_path:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("results", "training_history.png"))
    plt.close()

    return history