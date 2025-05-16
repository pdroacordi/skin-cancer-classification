from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout

class FixedDropout(Dropout):
    """Alias so that `load_model` can deserialize older EfficientNet weights."""
    pass

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
WEIGHTS   = Path(r"D:/PIBIC/python/skincancer/skincancer/results/unet_hair_remover/unet_hairmask.h5")
THRESH    = 0.01
IMG_SIZE  = 256
# filtros
_DIFF_RGB = 20
_MIN_LEN  = 20
_MAX_W    = 15

_NET = None  # cache global

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load():
    global _NET
    if _NET is None:
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"Peso {WEIGHTS} não encontrado.")
        _NET = load_model(WEIGHTS, compile=False, custom_objects={"FixedDropout": FixedDropout})
    return _NET


def _preprocess(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return img[None], (h, w)


def _postprocess(pred: np.ndarray, shape):
    h, w = shape
    mask = (pred[0, ..., 0] > THRESH).astype("uint8") * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _filter_color(mask: np.ndarray, bgr: np.ndarray, thr=_DIFF_RGB):
    diff = bgr.max(axis=2) - bgr.min(axis=2)
    mask[diff < thr] = 0
    return mask


def _filter_geom(mask: np.ndarray, min_len=_MIN_LEN, max_w=_MAX_W):
    if not mask.any():
        return mask
    sk = skeletonize(mask // 255).astype("uint8")
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    sk[dist < (max_w / 2)] = 0
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(sk, 8)
    keep = np.zeros_like(sk)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_len:
            keep[lbl == i] = 255
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def predict_hair_mask(bgr: np.ndarray) -> np.ndarray:
    """Mascara de cabelo uint8 0/255 no tamanho da imagem."""
    net = _load()
    x, shape = _preprocess(bgr)
    pred = net.predict(x, verbose=1)
    mask = _postprocess(pred, shape)
    mask = _filter_color(mask, bgr)
    mask = _filter_geom(mask)
    return mask