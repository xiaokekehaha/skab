from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """模型基类"""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        pass
        
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        pass
        
    @abstractmethod
    def set_params(self, **params) -> None:
        """设置参数"""
        pass 