import numpy as np

from ..base.preprocessor import ImagePreprocessor
from ..hair_removal.config import HairRemovalConfig
from ..hair_removal.inference import HairRemover


class HairRemovalStep(ImagePreprocessor):
    _shared_remover = None

    def __init__(self):
        if HairRemovalStep._shared_remover is None:
            cfg = HairRemovalConfig()
            HairRemovalStep._shared_remover = HairRemover(
                model_path=str(cfg.model_weights),
                use_tta=cfg.tta
            )
        self.remover = HairRemovalStep._shared_remover

    def process(self, image: np.ndarray) -> np.ndarray:
        clean, _ = self.remover.remove_hair(
            image_bgr=image,
            threshold=0.5,
            inpaint_radius=5
        )
        return clean

    def get_name(self) -> str:
        return "AI Hair Removal"
