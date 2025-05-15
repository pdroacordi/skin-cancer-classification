"""
Enhanced graphic preprocessing module with improved segmentation techniques.
This module integrates with the existing pipeline structure.
"""

from __future__ import annotations

import cv2
import numpy as np

def apply_graphic_preprocessing(image: np.ndarray,
                                use_hair_removal: bool = True,
                                use_contrast_enhancement: bool = True,
                                use_segmentation: bool = False,
                                visualize: bool = False
):
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

    processed = image.copy()
    mask = None
    # 1) Hair removal -------------------------------------------------------
    if use_hair_removal:
        processed, mask = _remove_hair(processed)

    # 2) Contrast enhancement (simple CLAHE + gamma) ------------------------
    if use_contrast_enhancement:
        processed = _enhance_contrast(processed)

    # 4) Visualize ----------------------------------------------------------
    if visualize:
        _visualize(image, processed, mask)

    return processed

def _enhance_contrast(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    mean = np.mean(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)) / 255.0
    gamma = 0.9 if mean < 0.5 else 1.1
    out = np.power(out / 255.0, gamma) * 255.0
    return out.astype(np.uint8)

def _remove_hair(bgr: np.ndarray):
    _dilate_kernel  = 5
    _inpaint_radius = 5

    try:
        import importlib
        unet_mod = importlib.import_module("hair_remover")
    except ImportError:
        import importlib.util, sys, pathlib
        local = pathlib.Path(__file__).with_name("hair_remover.py")
        if not local.exists():
            raise
        spec = importlib.util.spec_from_file_location("hair_remover", str(local))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        sys.modules["hair_remover"] = mod
        unet_mod = mod
    mask = unet_mod.predict_hair_mask(bgr)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_dilate_kernel, _dilate_kernel))
    mask_d = cv2.dilate(mask, k, iterations=1)
    cleaned = cv2.inpaint(bgr, mask_d, _inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned, mask_d


def _visualize(orig: np.ndarray, proc: np.ndarray, mask: np.ndarray | None = None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis("off")
    if mask is not None:
        overlay = orig.copy()
        overlay[mask == 255] = (0, 255, 0)  # verde na máscara
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Hair mask")
        plt.axis("off")
        col = 3
    else:
        col = 2
    plt.subplot(1, 3, col); plt.imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)); plt.title("Processed"); plt.axis("off")
    plt.tight_layout()
    plt.show()