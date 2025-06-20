import cv2
import numpy as np

from .model import create_chimeranet


class HairRemover:
    """Enhanced hair remover with better TTA and pipeline integration"""

    def __init__(self, model_path: str, use_tta: bool = True):
        """
        Initialize enhanced hair remover.

        Args:
            model_path: Path to trained model (.h5 file)
            use_tta: Whether to use test-time augmentation
        """
        self.use_tta = use_tta
        self.img_size = 448

        self.model = create_chimeranet(self.img_size)
        self.model.load_weights(model_path)

    def _predict_with_tta(self, img):
        """Enhanced TTA with more augmentations"""
        # Original
        preds = [self.model.predict(img[None], verbose=0)[0, :, :, 0]]

        # Horizontal flip
        img_hflip = np.flip(img, axis=1)
        pred_hflip = self.model.predict(img_hflip[None], verbose=0)[0, :, :, 0]
        preds.append(np.flip(pred_hflip, axis=1))

        # Vertical flip
        img_vflip = np.flip(img, axis=0)
        pred_vflip = self.model.predict(img_vflip[None], verbose=0)[0, :, :, 0]
        preds.append(np.flip(pred_vflip, axis=0))

        # 90 degree rotations
        for k in [1, 2, 3]:
            img_rot = np.rot90(img, k)
            pred_rot = self.model.predict(img_rot[None], verbose=0)[0, :, :, 0]
            preds.append(np.rot90(pred_rot, -k))

        # Average predictions
        return np.mean(preds, axis=0)

    def remove_hair(self, image_bgr, threshold=0.5, inpaint_radius=7):
        """
        Remove hair from dermoscopic image.

        Args:
            image_bgr: Input image in BGR format
            threshold: Threshold for hair mask binarization
            inpaint_radius: Radius for inpainting

        Returns:
            tuple: (hair_removed_image, hair_mask)
        """
        h, w = image_bgr.shape[:2]

        # Preprocess image
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Predict hair mask
        if self.use_tta:
            mask_pred = self._predict_with_tta(img_normalized)
        else:
            mask_pred = self.model.predict(img_normalized[None], verbose=0)[0, :, :, 0]

        # Post-process mask
        mask_binary = (mask_pred > threshold).astype(np.uint8) * 255

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

        # Resize mask back to original size
        mask_final = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        # Dilate mask slightly for better inpainting
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_dilated = cv2.dilate(mask_final, kernel_dilate, iterations=1)

        # Inpaint using advanced method
        hair_removed = cv2.inpaint(image_bgr, mask_dilated, inpaint_radius, cv2.INPAINT_TELEA)

        # Optional: Blend edges for smoother transition
        mask_float = mask_dilated.astype(np.float32) / 255.0
        mask_blur = cv2.GaussianBlur(mask_float, (5, 5), 0)
        mask_3ch = np.stack([mask_blur] * 3, axis=-1)

        hair_removed = (hair_removed * mask_3ch + image_bgr * (1 - mask_3ch)).astype(np.uint8)

        return hair_removed, mask_final
