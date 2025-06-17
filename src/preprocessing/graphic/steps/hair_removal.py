import numpy as np

from base.preprocessor import ImagePreprocessor
from ..hair_removal.config import HairRemovalConfig
from ..hair_removal.inference import HairRemover


class HairRemovalStep(ImagePreprocessor):
    def __init__(self):
        self.remover = HairRemover(
            HairRemovalConfig()
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        result, _ = self.remover.remove_hair(image)
        return result

    def get_name(self) -> str:
        return "AI Hair Removal"
