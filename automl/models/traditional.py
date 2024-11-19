from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from automl.base.base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import pmdarima as pm
import pandas as pd
import numpy as np
from typing import Dict, Any

class SKLearnModel(BaseModel):
    """封装 scikit-learn 模型的基类"""
    def __init__(self, model):
        self.model = model
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
        
    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()
        
    def set_params(self, **params) -> None:
        self.model.set_params(**params)

class RandomForestModel(SKLearnModel):
    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor(**kwargs))

class LinearModel(SKLearnModel):
    def __init__(self, **kwargs):
        super().__init__(LinearRegression(**kwargs)) 

class ARIMAModel(BaseModel):
    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        self.p = p
        self.d = d
        self.q = q
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = ARIMA(y, order=(self.p, self.d, self.q))
        self.model = self.model.fit()
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.forecast(steps=len(X))
        
    def get_params(self) -> Dict[str, Any]:
        return {'p': self.p, 'd': self.d, 'q': self.q}
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)

class AutoARIMAModel(BaseModel):
    def __init__(self, seasonal: bool = True, m: int = 12):
        self.seasonal = seasonal
        self.m = m
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = pm.auto_arima(
            y,
            seasonal=self.seasonal,
            m=self.m,
            suppress_warnings=True,
            error_action="ignore"
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(n_periods=len(X))

class ProphetModel(BaseModel):
    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        df = pd.DataFrame({'ds': X.index, 'y': y})
        self.model.fit(df)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        future = pd.DataFrame({'ds': X.index})
        forecast = self.model.predict(future)
        return forecast['yhat'].values

class ExponentialSmoothingModel(BaseModel):
    def __init__(self, seasonal: str = 'add', seasonal_periods: int = 12):
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = ExponentialSmoothing(
            y,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods
        ).fit()
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.forecast(len(X)) 