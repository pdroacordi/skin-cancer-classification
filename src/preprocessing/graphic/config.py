from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    use_hair_removal: bool = True
    use_contrast_enhancement: bool = True
    use_color_normalization: bool = False
    color_norm_stats_path: Optional[str] = None  # path to fitted Reinhard stats
    confidence_threshold: float = 0.65
    inpaint_radius: int = 5
    morphological_kernel_size: int = 5
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)