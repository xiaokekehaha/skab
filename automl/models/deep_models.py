import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from automl.base.base_model import BaseModel
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
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

from typing import Dict, Any
from ..base.base_model import BaseModel

class TimeSeriesTransformer(BaseModel):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.model = TransformerModel(input_dim, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            epochs: int = 100, 
            batch_size: int = 32,
            lr: float = 0.001) -> None:
        
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        y_tensor = torch.FloatTensor(y.values).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()
        
    def get_params(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            # 其他参数...
        }
    
    def set_params(self, **params) -> 'TimeSeriesTransformer':
        for key, value in params.items():
            setattr(self, key, value)
        return self

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding='same', dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding='same', dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCNModel(BaseModel):
    def __init__(self, input_dim: int, num_channels: list, kernel_size: int):
        super().__init__()
        self.model = TCN(input_dim, num_channels, kernel_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            epochs: int = 100,
            batch_size: int = 32,
            lr: float = 0.001) -> None:
        # 实现类似 TimeSeriesTransformer 的训练逻辑
        pass
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 实现类似 TimeSeriesTransformer 的预测逻辑
        pass

class LSTMModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            epochs: int = 100,
            batch_size: int = 32,
            lr: float = 0.001) -> None:
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        y_tensor = torch.FloatTensor(y.values).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
                
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.eval()
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        with torch.no_grad():
            predictions = self(X_tensor)
        return predictions.cpu().numpy()

class GRUModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        predictions = self.linear(gru_out[:, -1, :])
        return predictions
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            epochs: int = 100,
            batch_size: int = 32,
            lr: float = 0.001) -> None:
        # 实现类似 LSTM ���训练逻辑
        pass
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 实现类似 LSTM 的预测逻辑
        pass

class Seq2SeqModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x, target_len: int = 1):
        # Encoder
        _, (hidden, cell) = self.encoder(x)
        
        # Decoder
        decoder_input = torch.zeros(x.size(0), 1, 1).to(self.device)
        outputs = []
        
        for _ in range(target_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.linear(decoder_output)
            outputs.append(prediction)
            decoder_input = prediction
            
        return torch.cat(outputs, dim=1)