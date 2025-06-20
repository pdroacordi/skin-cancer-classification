from dataclasses import dataclass
from pathlib import Path


@dataclass
class HairRemovalConfig:
    img_size: int = 448
    batch_size: int = 4
    epochs: int = 600
    initial_lr = 2e-4
    min_lr = 1e-7
    model_weights = Path(__file__).parents[4] / "results" / "chimera" / "best_weights.h5"
    tta = True
    use_mixed_precision = True
    gradient_clip_norm = 1.0

    # Data augmentation parameters
    augmentation_prob = 0.8
    rotation_range = 15
    zoom_range = (0.9, 1.1)
    brightness_range = (0.9, 1.1)
    contrast_range = (0.9, 1.1)

    # Loss weights (will be adjusted during training)
    loss_weights = {
        'bce': 0.3,
        'dice': 0.4,
        'tversky': 0.3
    }
