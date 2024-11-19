from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from automl.base.base_model import BaseModel
import copy

class EnsembleModel(BaseModel):
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for model in self.models:
            model.fit(X, y)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(weight * pred)
        return np.sum(predictions, axis=0)
        
    def get_params(self) -> Dict[str, Any]:
        return {
            'models': [model.get_params() for model in self.models],
            'weights': self.weights
        }
        
    def set_params(self, **params) -> None:
        if 'weights' in params:
            self.weights = params['weights'] 

class StackingModel(BaseModel):
    def __init__(self, 
                 base_models: List[BaseModel],
                 meta_model: BaseModel,
                 n_splits: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # 训练基础模型
        base_predictions = np.zeros((len(X), len(self.base_models)))
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            base_predictions[:, i] = model.predict(X).flatten()
            
        # 训练元模型
        self.meta_model.fit(pd.DataFrame(base_predictions), y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 获取基础模型预测
        base_predictions = np.zeros((len(X), len(self.base_models)))
        for i, model in enumerate(self.base_models):
            base_predictions[:, i] = model.predict(X).flatten()
            
        # 使用元模型进行最终预测
        return self.meta_model.predict(pd.DataFrame(base_predictions))

class BaggingModel(BaseModel):
    def __init__(self, 
                 base_model: BaseModel,
                 n_estimators: int = 10,
                 sample_size: float = 0.8):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.models = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.models = []
        sample_size = int(len(X) * self.sample_size)
        
        for _ in range(self.n_estimators):
            # 随机采样
            indices = np.random.choice(len(X), sample_size, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # 训练新模型
            model = copy.deepcopy(self.base_model)
            model.fit(X_sample, y_sample)
            self.models.append(model)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X).flatten()
        return np.mean(predictions, axis=1)