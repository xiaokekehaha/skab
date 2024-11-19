from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import numpy as np

class BaseOptimizer(ABC):
    """优化器基类"""
    
    @abstractmethod
    def optimize(self, 
                objective: Callable,
                param_space: Dict[str, Any],
                n_trials: int = 100) -> Dict[str, Any]:
        """优化超参数"""
        pass 