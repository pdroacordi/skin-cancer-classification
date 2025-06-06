"""Centralised constants so we keep magic strings out of the logic."""

from pathlib import Path
from typing import List, Dict

# Model groups ---------------------------------------------------------------
CNN_MODELS: List[str] = ["VGG19", "Inception", "ResNet", "Xception"]
ML_CLASSIFIERS: List[str] = ["RandomForest", "XGBoost", "AdaBoost", "ExtraTrees"]

ALG_NICE = dict(
    adaboost="AdaBoost",
    extratrees="ExtraTrees",
    randomforest="RandomForest",
    xgboost="XGBoost",
)
NET_NICE = dict(
    Resnet="ResNet",
    Vgg19="VGG19",
)

# Colours --------------------------------------------------------------------
COLOR_PALETTE: Dict[str, str] = {
    # meta
    "train": "#ff7f0e",  # orange
    "test": "#2ca02c",   # green
    # cnn backbones
    "VGG19": "#1f77b4",
    "Inception": "#d62728",
    "ResNet": "#9467bd",
    "Xception": "#8c564b",
    # classical ML learners
    "RandomForest": "#17becf",
    "XGBoost": "#bcbd22",
    "AdaBoost": "#ff9500",
    "ExtraTrees": "#e377c2",
}

CLASSES: List[str] = [
    "akiec",  # Actinic keratoses and intra‑epithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis‑like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc",   # Vascular lesions
]

# Figures --------------------------------------------------------------------
DEFAULT_FIGSIZE = (14, 8)
DEFAULT_DPI = 300
OUTPUT_DIR = Path("../../figures")  # changed via CLI flag if needed
OUTPUT_DIR.mkdir(exist_ok=True)
HEATMAP_CMAP = "YlGnBu"