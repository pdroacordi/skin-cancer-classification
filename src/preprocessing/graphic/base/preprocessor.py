from abc import ABC, abstractmethod

import numpy as np


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessing steps."""

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process the image and return the result."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the preprocessing step."""
        pass