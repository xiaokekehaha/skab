import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, Optional, Tuple
import optuna
import pandas as pd

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.decoder(x)

class TCN(nn.Module):
    def __init__(self, input_dim: int, num_channels: List[int], kernel_size: int):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation)
            )
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        x = self.network(x)
        return self.linear(x[:, :, -1])

class AutoTSPredictor:
    def __init__(
        self,
        models: List[str] = ['transformer', 'tcn', 'lstm', 'gru'],
        metric: str = 'rmse',
        cv_splits: int = 5,
        max_trials: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.models = models
        self.metric = metric
        self.cv = TimeSeriesSplit(n_splits=cv_splits)
        self.max_trials = max_trials
        self.device = device
        self.best_model = None
        self.best_params = None
        self.scaler = StandardScaler()
        
    def optimize(self, X: pd.DataFrame, y: pd.Series):
        """自动优化模型选择和超参数"""
        study = optuna.create_study(direction='minimize')
        
        def objective(trial):
            # 选择模型
            model_name = trial.suggest_categorical('model', self.models)
            
            # 根据不同模型类型设置超参数搜索空间
            params = self._get_model_params(trial, model_name)
            
            # 交叉验证评估
            scores = []
            for train_idx, val_idx in self.cv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = self._build_model(model_name, params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                score = mean_squared_error(y_val, y_pred, squared=False)
                scores.append(score)
                
            return np.mean(scores)
            
        study.optimize(objective, n_trials=self.max_trials)
        
        self.best_model = self._build_model(
            study.best_params['model'],
            self._get_model_params(study.best_trial, study.best_params['model'])
        )
        self.best_params = study.best_params
        
    def _get_model_params(self, trial: optuna.Trial, model_name: str) -> Dict:
        """扩展的超参数搜索空间"""
        if model_name == 'transformer':
            return {
                'd_model': trial.suggest_int('d_model', 32, 128),
                'nhead': trial.suggest_int('nhead', 2, 8),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'batch_size': trial.suggest_int('batch_size', 32, 128),
                'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2)
            }
        elif model_name == 'tcn':
            return {
                'num_channels': [trial.suggest_int(f'channel_{i}', 32, 128) 
                               for i in range(trial.suggest_int('num_levels', 2, 4))],
                'kernel_size': trial.suggest_int('kernel_size', 2, 5),
                'batch_size': trial.suggest_int('batch_size', 32, 128),
                'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2)
            }
        elif model_name == 'lstm':
            return {
                'units': trial.suggest_int('units', 32, 256),
                'layers': trial.suggest_int('layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            }
    
    def _build_model(self, model_name: str, params: Dict, input_dim: int):
        """扩展的模型构建"""
        if model_name == 'transformer':
            return TimeSeriesTransformer(
                input_dim=input_dim,
                d_model=params['d_model'],
                nhead=params['nhead'],
                num_layers=params['num_layers']
            ).to(self.device)
        elif model_name == 'tcn':
            return TCN(
                input_dim=input_dim,
                num_channels=params['num_channels'],
                kernel_size=params['kernel_size']
            ).to(self.device)
        elif model_name == 'lstm':
            return self._build_lstm_model(params)
    
    def _build_lstm_model(self, params: Dict):
        """构建LSTM模型"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential()
        for i in range(params['layers']):
            if i == 0:
                model.add(LSTM(params['units'], return_sequences=True if params['layers'] > 1 else False))
            elif i == params['layers'] - 1:
                model.add(LSTM(params['units']))
            else:
                model.add(LSTM(params['units'], return_sequences=True))
            model.add(Dropout(params['dropout']))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """创建时序序列"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def _train_epoch(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def fit(self, X: pd.DataFrame, y: pd.Series, seq_length: int = 10):
        """增强的训练流程"""
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq = self._create_sequences(X_scaled, seq_length)
        
        # 优化和训练
        if self.best_model is None:
            self.optimize(X_seq, y_seq)
            
        # 使用最佳参数重新训练
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_seq),
            torch.FloatTensor(y_seq)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.best_params.get('batch_size', 32),
            shuffle=True
        )
        
        optimizer = torch.optim.Adam(
            self.best_model.parameters(),
            lr=self.best_params.get('lr', 0.001)
        )
        criterion = nn.MSELoss()
        
        for epoch in range(100):  # 可以添加早停
            train_loss = self._train_epoch(
                self.best_model,
                train_loader,
                optimizer,
                criterion
            )
            
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """增强的预测功能"""
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.best_model.eval()
        with torch.no_grad():
            predictions = self.best_model(X_tensor)
            
        return predictions.cpu().numpy() 