from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np


class BalancingStrategy(ABC):
    """Abstract base class for balancing strategies."""

    @abstractmethod
    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance the dataset."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get balancing parameters."""
        pass