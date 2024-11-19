from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeSeriesProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_sequences(self, 
                        data: np.ndarray,
                        seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """创建时序序列"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
        
    def preprocess(self, 
                  df: pd.DataFrame,
                  seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """预处理数据"""
        scaled_data = self.scaler.fit_transform(df)
        return self.create_sequences(scaled_data, seq_length) 