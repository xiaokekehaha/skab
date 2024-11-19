# AutoML 时序预测框架

## 项目结构 

## 主要功能

### 1. 深度学习模型 (deep_models.py)

- **TimeSeriesTransformer**
  - Transformer架构的时序预测模型
  - 支持位置编码和多头注意力
  - 可配置模型深度和维度

- **TCN (Temporal Convolutional Network)**
  - 因果卷积网络
  - 支持空洞卷积
  - 可配置通道数和核大小

- **LSTM**
  - 长短期记忆网络
  - 支持多层和dropout
  - 梯度裁剪防止梯度爆炸

- **GRU**
  - 门控循环单元
  - 类似LSTM但参数更少
  - 适合中等规模数据

- **Seq2Seq**
  - 编码器-解码器架构
  - 支持多步预测
  - Teacher forcing训练

### 2. 传统模型 (traditional.py)

- **ARIMA**
  - 自回归积分滑动平均模型
  - 支持季节性
  - 自动参数选择

- **Prophet**
  - Facebook的时序预测模型
  - 处理节假日效应
  - 自动处理缺失值

- **ExponentialSmoothing**
  - 指数平滑模型
  - 支持趋势和季节性
  - 适合短期预测

- **RandomForest**
  - 随机森林回归
  - 集成学习方法
  - 处理非线性关系

### 3. 集成模型 (ensemble.py)

- **EnsembleModel**
  - 基础集成模型
  - 支持加权平均
  - 可配置权重

- **StackingModel**
  - 堆叠集成
  - 元学习器
  - 交叉验证训练

- **BaggingModel**
  - Bootstrap聚合
  - 降低方差
  - 并行训练支持

## 核心特性

1. **模型选择**
   - 自动模型选择
   - 超参数优化
   - 交叉验证评估

2. **数据处理**
   - 自动缩放
   - 序列创建
   - 缺失值处理

3. **训练优化**
   - GPU加速支持
   - 早停机制
   - 学习率调度

4. **评估指标**
   - RMSE
   - MAE
   - 自定义指标支持

## 最近更新

1. 添加了新的深度学习模型:
   - GRU
   - Seq2Seq
   - TCN完整实现

2. 增强了传统模型:
   - 添加AutoARIMA
   - 完善Prophet接口
   - 添加ExponentialSmoothing

3. 改进了集成方法:
   - 添加Stacking
   - 添加Bagging
   - 优化权重分配

4. 优化了训练流程:
   - 添加梯度裁剪
   - 实现学习率调度
   - 改进早停机制

## 待办事项

1. 模型扩展:
   - [ ] 添加VAE模型
   - [ ] 实现Attention-LSTM
   - [ ] 添加XGBoost支持

2. 功能增强:
   - [ ] 特征选择
   - [ ] 在线学习
   - [ ] 模型解释性

3. 性能优化:
   - [ ] 分布式训练
   - [ ] 模型压缩
   - [ ] 推理加速

4. 其他改进:
   - [ ] 完善文档
   - [ ] 添加单元测试
   - [ ] 示例notebook