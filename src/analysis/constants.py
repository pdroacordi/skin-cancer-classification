"""Centralised constants so we keep magic strings out of the logic."""

from pathlib import Path
from typing import List, Dict

# Model groups ---------------------------------------------------------------
CNN_MODELS: List[str] = ["Inception", "Xception", "ConvNeXt", "EfficientNetV2S"]
ML_CLASSIFIERS: List[str] = [
    "RandomForest", "XGBoost", "LightGBM", "HistGradientBoosting", "ExtraTrees", "SVM"
]

ALG_NICE = dict(
    extratrees="ExtraTrees",
    randomforest="RandomForest",
    xgboost="XGBoost",
    lightgbm="LightGBM",
    histgradientboosting="HistGradientBoosting",
    svm="SVM",
    dynamic_ensemble="DynamicEnsemble",
)
NET_NICE = dict(
    Convnext="ConvNeXt",
    Efficientnetv2s="EfficientNetV2S",
)

# Colours --------------------------------------------------------------------
COLOR_PALETTE: Dict[str, str] = {
    # meta
    "train": "#ff7f0e",  # orange
    "test": "#2ca02c",   # green
    # cnn backbones
    "Inception": "#d62728",
    "Xception": "#8c564b",
    "ConvNeXt": "#9467bd",
    "EfficientNetV2S": "#1f77b4",
    # classical ML learners
    "RandomForest": "#17becf",
    "XGBoost": "#bcbd22",
    "LightGBM": "#aec7e8",
    "HistGradientBoosting": "#ffbb78",
    "ExtraTrees": "#e377c2",
    "SVM": "#98df8a",
    "DynamicEnsemble": "#ff9500",
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
HEATMAP_CMAP = "YlGnBu"