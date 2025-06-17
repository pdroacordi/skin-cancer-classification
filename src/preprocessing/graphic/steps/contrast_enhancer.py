import cv2
import numpy as np

from preprocessing.graphic.base.preprocessor import ImagePreprocessor
from preprocessing.graphic.config import PreprocessingConfig


class ContrastEnhancer(ImagePreprocessor):
    """Enhanced contrast adjustment using CLAHE and adaptive gamma correction."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_size
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement to the image."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        l_channel = self.clahe.apply(l_channel)

        # Merge channels
        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Apply adaptive gamma correction
        enhanced = self._apply_adaptive_gamma(enhanced)

        return enhanced

    def _apply_adaptive_gamma(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive gamma correction based on image brightness."""
        # Calculate mean brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0

        # Determine gamma value
        if mean_brightness < 0.3:
            gamma = 0.7  # Brighten dark images more
        elif mean_brightness < 0.5:
            gamma = 0.9
        elif mean_brightness > 0.7:
            gamma = 1.2  # Darken bright images
        else:
            gamma = 1.1

        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(256)]).astype("uint8")

        return cv2.LUT(image, table)

    def get_name(self) -> str:
        return "Contrast Enhancement"