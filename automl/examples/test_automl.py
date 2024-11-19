import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.deep_models import (
    TimeSeriesTransformer, 
    TCNModel,
    LSTMModel, 
    GRUModel,
    Seq2SeqModel
)
from models.traditional import (
    ARIMAModel,
    AutoARIMAModel,
    ProphetModel,
    ExponentialSmoothingModel
)
from models.ensemble import (
    EnsembleModel,
    StackingModel,
    BaggingModel
)
from utils.data_processor import TimeSeriesProcessor

def generate_sample_data(n_samples=1000):
    """生成测试用的时序数据"""
    np.random.seed(42)
    t = np.linspace(0, 100, n_samples)
    
    # 生成趋势、季节性和噪声
    trend = 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50)
    noise = np.random.normal(0, 1, n_samples)
    
    y = trend + seasonal + noise
    X = pd.DataFrame({
        'time': t,
        'feature1': np.sin(2 * np.pi * t / 30),
        'feature2': np.cos(2 * np.pi * t / 60)
    })
    
    return X, pd.Series(y, name='target')

def test_single_model(model, X_train, y_train, X_test, y_test, model_name):
    """测试单个模型"""
    print(f"\nTesting {model_name}...")
    
    try:
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return y_pred, mse, mae
    
    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")
        return None, None, None

def test_ensemble_models(X_train, y_train, X_test, y_test):
    """测试集成模型"""
    # 创建基础模型
    base_models = [
        LSTMModel(input_dim=X_train.shape[1]),
        TCNModel(input_dim=X_train.shape[1], num_channels=[32, 32], kernel_size=3),
        ProphetModel()
    ]
    
    # 测试简单集成
    ensemble = EnsembleModel(base_models)
    test_single_model(ensemble, X_train, y_train, X_test, y_test, "Ensemble")
    
    # 测试Stacking
    meta_model = LSTMModel(input_dim=len(base_models))
    stacking = StackingModel(base_models, meta_model)
    test_single_model(stacking, X_train, y_train, X_test, y_test, "Stacking")
    
    # 测试Bagging
    bagging = BaggingModel(LSTMModel(input_dim=X_train.shape[1]))
    test_single_model(bagging, X_train, y_train, X_test, y_test, "Bagging")

def main():
    # 生成数据
    X, y = generate_sample_data()
    
    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 数据预处理
    processor = TimeSeriesProcessor()
    X_train_seq, y_train_seq = processor.preprocess(X_train)
    X_test_seq, y_test_seq = processor.preprocess(X_test)
    
    # 测试深度学习模型
    models = {
        'Transformer': TimeSeriesTransformer(input_dim=X_train.shape[1]),
        'TCN': TCNModel(input_dim=X_train.shape[1], num_channels=[32, 32], kernel_size=3),
        'LSTM': LSTMModel(input_dim=X_train.shape[1]),
        'GRU': GRUModel(input_dim=X_train.shape[1]),
        'Seq2Seq': Seq2SeqModel(input_dim=X_train.shape[1])
    }
    
    results = {}
    for name, model in models.items():
        y_pred, mse, mae = test_single_model(model, X_train_seq, y_train_seq, 
                                           X_test_seq, y_test_seq, name)
        results[name] = {'mse': mse, 'mae': mae, 'predictions': y_pred}
    
    # 测试传统模型
    traditional_models = {
        'ARIMA': ARIMAModel(),
        'AutoARIMA': AutoARIMAModel(),
        'Prophet': ProphetModel(),
        'ExponentialSmoothing': ExponentialSmoothingModel()
    }
    
    for name, model in traditional_models.items():
        y_pred, mse, mae = test_single_model(model, X_train, y_train, X_test, y_test, name)
        results[name] = {'mse': mse, 'mae': mae, 'predictions': y_pred}
    
    # 测试集成模型
    test_ensemble_models(X_train, y_train, X_test, y_test)
    
    # 打印总结
    print("\nSummary of Results:")
    for name, metrics in results.items():
        if metrics['mse'] is not None:
            print(f"\n{name}:")
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    main() 