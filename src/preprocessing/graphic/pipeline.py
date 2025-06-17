"""
Enhanced graphic preprocessing module with state-of-the-art hair removal.
Implements modern deep learning techniques for dermoscopic image preprocessing.
"""

from __future__ import annotations

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from base.preprocessor import ImagePreprocessor
from config import PreprocessingConfig
from steps.contrast_enhancer import ContrastEnhancer
from steps.hair_removal import HairRemovalStep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Main preprocessing pipeline that orchestrates all preprocessing steps."""

    def __init__(self, cfg: PreprocessingConfig):
        self.cfg = cfg
        self.steps: list[ImagePreprocessor] = []
        self._build()

    def _build(self):
        if self.cfg.use_hair_removal:
            self.steps.append(HairRemovalStep())
        if self.cfg.use_contrast_enhancement:
            self.steps.append(ContrastEnhancer(self.cfg))

    def process(self, img: np.ndarray, visualize: bool = False) -> np.ndarray:
        out = img.copy()
        for step in self.steps:
            logger.info(f"Applying {step.get_name()}")
            out = step.process(out)
            if visualize:
                self._viz(img, out, step.get_name())
        return out

    def _viz(self, orig, proc, name):
        import matplotlib.pyplot as plt
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
        a.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)); a.set_title("Original"); a.axis("off")
        b.imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)); b.set_title(name); b.axis("off")
        plt.show()



# Convenience function matching original API
def apply_graphic_preprocessing(image: np.ndarray,
                              use_hair_removal: bool = True,
                              use_contrast_enhancement: bool = True,
                              use_segmentation: bool = False,
                              visualize: bool = False) -> np.ndarray:
    """
    Apply graphic preprocessing to dermoscopic image.

    Args:
        image: Input BGR image
        use_hair_removal: Whether to apply hair removal
        use_contrast_enhancement: Whether to enhance contrast
        use_segmentation: Whether to segment the lesion
        visualize: Whether to visualize intermediate steps

    Returns:
        Preprocessed image
    """
    config = PreprocessingConfig(
        use_hair_removal=use_hair_removal,
        use_contrast_enhancement=use_contrast_enhancement
    )

    pipeline = PreprocessingPipeline(config)
    return pipeline.process(image, visualize=visualize)