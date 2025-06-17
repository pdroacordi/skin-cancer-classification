from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class HairRemovalConfig:
    img_size: int = 448
    batch_size: int = 2
    epochs: int = 300
    lr: float = 1e-4
    model_weights: Optional[Path] = None
    tta: bool = True
