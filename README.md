# 基于深度学习的分布式光伏发电不确定性建模

本项目面向论文方向“基于深度学习的分布式光伏发电不确定性建模技术研究”，以 Plant 1 光伏电站发电数据和气象传感器数据为实验对象，研究 15 分钟时间粒度下未来 4 小时光伏出力的概率预测问题。

项目采用 PyTorch 实现改进贝叶斯神经网络（Improved Bayesian PV Network）。模型以全连接神经网络处理气象和周期时间特征，以一维卷积神经网络处理历史出力与历史气象序列，并将预测点前一时刻的强相关变量引入贝叶斯概率融合层，最终通过 `BayesianLinear` 输出预测均值和预测方差，从而同时给出点预测结果和不确定性区间。

## 目录

- [1. 研究任务](#1-研究任务)
- [2. 数据与预处理](#2-数据与预处理)
- [3. 特征设计](#3-特征设计)
- [4. 模型方法](#4-模型方法)
- [5. 训练目标与不确定性估计](#5-训练目标与不确定性估计)
- [6. 实验设置](#6-实验设置)
- [7. 评价指标](#7-评价指标)
- [8. 复现实验流程](#8-复现实验流程)
- [9. 输出结果说明](#9-输出结果说明)
- [10. 配置文件说明](#10-配置文件说明)
- [11. 代码结构](#11-代码结构)
- [12. 常见问题](#12-常见问题)
- [13. 论文写作参考](#13-论文写作参考)
- [附录：术语与英文缩写说明](#附录术语与英文缩写说明)

## 1. 研究任务

本项目研究的是分布式光伏电站的超短期概率预测任务。给定过去一段时间内的电站出力、辐照度、温度等信息，模型需要预测未来多个时间步的交流功率 `AC_POWER`。

当前实验设置如下：

| 项目 | 设置 |
| --- | --- |
| 数据集 | Plant 1 发电数据与气象传感器数据 |
| 时间粒度 | 15 分钟 |
| 输入窗口 | 过去 32 步，即 8 小时 |
| 预测窗口 | 未来 16 步，即 4 小时 |
| 预测目标 | 电站级 `AC_POWER` |
| 数据切分 | 按时间顺序划分 train / val / test |
| 主要模型 | `ImprovedBayesianPVNet` |
| 主要目标 | 同时获得点预测精度和概率预测区间 |

与单纯点预测不同，本项目关注预测不确定性。模型不仅输出未来功率均值，还输出预测方差，并通过 Monte Carlo 多次前向传播构造预测区间。

## 2. 数据与预处理

原始数据放置在 `dataset/` 目录：

```text
dataset/
├── Plant_1_Generation_Data.csv
└── Plant_1_Weather_Sensor_Data.csv
```

预处理逻辑位于 `src/data.py`、`src/features.py` 和 `src/dataset.py`。主流程会自动完成以下步骤：

1. 解析 `DATE_TIME` 时间列。
2. 将多个逆变器记录聚合为电站级时间序列。
3. 按时间合并气象传感器数据。
4. 补齐 15 分钟时间索引，并对缺失值进行处理。
5. 构造周期时间特征、白天标记和最近时刻直接输入特征。
6. 按时间顺序划分训练集、验证集和测试集。
7. 在各数据子集内部构造滑动窗口，避免窗口跨越数据切分边界。
8. 只在训练集上拟合 scaler，再用于验证集和测试集，避免标准化阶段的数据泄漏。

窗口构造方式为：

```text
history: [t - lookback, ..., t - 1]
target : [t, ..., t + horizon - 1]
```

当前默认配置为：

```text
过去 32 个 15 分钟点 -> 未来 16 个 15 分钟点
即 8h 历史信息 -> 4h 预测结果
```

需要注意：公开数据中的气象变量是实测气象，不是真实数值天气预报（NWP）。因此默认配置 `use_future_weather: false`，预测窗口内的天气输入使用最近观测气象的持久化值，避免把未来实测气象泄漏给模型。若后续接入真实 NWP，可再改为使用未来天气预报特征。

## 3. 特征设计

特征分组由 `src.features.split_feature_columns()` 统一维护。当前模型将输入拆分为四类：

| 分组 | 特征 | 作用 |
| --- | --- | --- |
| `history` | `AC_POWER`、`DC_POWER`、`IRRADIATION`、`AMBIENT_TEMPERATURE`、`MODULE_TEMPERATURE` | 表示过去 8 小时的出力和气象变化，输入 CNN 分支 |
| `weather` | `IRRADIATION`、`AMBIENT_TEMPERATURE`、`MODULE_TEMPERATURE` | 表示预测窗口天气条件，默认使用最近观测值持久化 |
| `time` | `hour_sin`、`hour_cos`、`dayofyear_sin`、`dayofyear_cos`、`month_sin`、`month_cos` | 表示日内、年内和月内周期性 |
| `direct` | `last_ac_power`、`last_dc_power`、`last_irradiation` | 表示预测点前一时刻的强相关变量 |

时间特征采用 `sin/cos` 周期编码，而不是直接使用小时、日期或月份数值。这样可以避免模型误认为 `23:45` 和 `00:00` 距离很远，更符合光伏出力的周期特征。

## 4. 模型方法

默认模型为 `ImprovedBayesianPVNet`，代码位于 `src/models/improved_bnn.py`。模型在输入端采用独立特征处理模块：气象特征与光伏出力之间存在复杂非线性关系，因此输入全连接神经网络模块；历史出力和历史气象数据具有较强时序性，因此输入一维卷积神经网络模块；预测点前一时刻的功率、辐照度等变量与预测目标相关性较强，因此通过直接输入分支引入后端贝叶斯概率融合层。

模型结构可概括为：

```text
多输入分支特征提取 -> 特征拼接 -> 贝叶斯全连接融合 -> 均值/方差双输出头
```

### 4.1 网络结构

```text
history: [batch, lookback, 5]
  └─ HistoryCNNBranch
     ├─ Conv1d(5 -> branch_dim, kernel_size=3, padding=1)
     ├─ ReLU
     ├─ AvgPool1d(kernel_size=2)
     ├─ Conv1d(branch_dim -> branch_dim, kernel_size=3, padding=1)
     ├─ ReLU
     ├─ AdaptiveAvgPool1d(1)
     └─ Linear(branch_dim -> branch_dim)

weather: [batch, horizon, 3]
  └─ SequenceMLPBranch
     ├─ Flatten
     ├─ Linear(3 * horizon -> hidden_dim)
     ├─ ReLU
     └─ Linear(hidden_dim -> branch_dim)

time: [batch, horizon, 6]
  └─ SequenceMLPBranch
     ├─ Flatten
     ├─ Linear(6 * horizon -> hidden_dim)
     ├─ ReLU
     └─ Linear(hidden_dim -> branch_dim)

direct: [batch, 3]
  └─ DirectInputBranch
     ├─ Linear(3 -> branch_dim)
     ├─ ReLU
     └─ Linear(branch_dim -> branch_dim / 2)

fusion:
  concat(history, weather, time, direct)
  ├─ BayesianLinear(fusion_dim -> hidden_dim)
  ├─ ReLU
  ├─ BayesianLinear(hidden_dim -> hidden_dim)
  ├─ ReLU
  ├─ mean_head: BayesianLinear(hidden_dim -> horizon)
  └─ log_var_head: BayesianLinear(hidden_dim -> horizon)
```

当前默认配置为：

```yaml
lookback: 32
horizon: 16
hidden_dim: 256
branch_dim: 64
```

因此融合层输入维度为：

```text
64(history) + 64(weather) + 64(time) + 32(direct) = 224
```

模型最终输出两个形状为 `[batch, horizon]` 的张量：

- `mean`：未来 16 个预测步的功率均值。
- `log_var`：未来 16 个预测步的对数方差，用于描述数据噪声不确定性。

### 4.2 网络层名称解释

| 名称 | 含义 | 在当前模型中的作用 |
| --- | --- | --- |
| `Conv1d` | 一维卷积层 | 沿时间轴提取历史出力、辐照度和温度序列的局部变化模式 |
| `ReLU` | Rectified Linear Unit，线性整流激活函数 | 引入非线性，使模型能学习复杂关系 |
| `AvgPool1d` | 一维平均池化 | 对历史序列做下采样，压缩时间长度并保留局部趋势 |
| `AdaptiveAvgPool1d(1)` | 自适应平均池化到长度 1 | 将任意长度的卷积输出压缩成固定长度表示 |
| `Flatten` | 展平操作 | 把 `[batch, horizon, features]` 形式的序列输入展开成向量 |
| `Linear` | 普通全连接层 | 用于确定性特征映射 |
| `BayesianLinear` | 贝叶斯全连接层 | 权重不是固定值，而是可采样的高斯分布，用于建模参数不确定性 |
| `mean_head` | 均值输出头 | 输出未来各预测步的功率均值 |
| `log_var_head` | 方差输出头 | 输出未来各预测步的对数方差 |

其中 `branch_dim` 表示每个输入分支提取出的特征表示维度，`hidden_dim` 表示融合后的贝叶斯全连接层宽度。可以把前者理解为“每类输入先压缩成多长的特征向量”，后者理解为“融合后用多大的隐藏层继续学习”。

### 4.3 模型设计动机

不同输入变量具有不同结构。历史出力和历史气象序列包含局部波动和趋势，适合使用 1D-CNN 提取时序局部模式；气象变量和周期时间特征与光伏出力之间存在复杂非线性关系，适合用 MLP/全连接网络进行特征映射；最近时刻出力、直流功率和辐照度与短期预测强相关，因此单独作为 direct 分支输入，并与其他分支特征共同进入贝叶斯概率融合层。

这种多分支结构相比直接拼接所有变量，可以更清晰地表达不同特征类型的物理含义，也便于后续进行消融实验。

## 5. 训练目标与不确定性估计

### 5.1 BayesianLinear

项目中的概率层为 `BayesianLinear`，代码位于 `src/models/bayesian_layers.py`。普通 `Linear` 层的权重是确定值，而 `BayesianLinear` 将权重和偏置视为高斯变分后验：

```text
w = mu + softplus(rho) * eps
eps ~ Normal(0, 1)
```

每次 forward 都会通过重参数化采样不同权重，因此同一输入可以得到不同预测结果。模型通过这种方式刻画参数不确定性。

### 5.2 ELBO 风格损失

训练损失位于 `src/losses.py`，由高斯负对数似然和 KL 正则组成：

```text
loss = GaussianNLL(mean, log_var, target)
       + kl_beta * KL(q(w) || p(w))
```

其中：

- `GaussianNLL` 约束预测均值和预测方差，使模型学习数据噪声。
- `KL(q(w) || p(w))` 约束贝叶斯层后验分布不要过度偏离先验。
- `kl_beta` 控制 KL 项权重，当前最优调参结果为 `1.2114960587728605e-05`。

### 5.3 Monte Carlo 推理

推理阶段使用 `src.predict.mc_predict()` 执行多次前向传播。每次前向传播都会采样一组贝叶斯层权重，得到一组预测均值和方差。最终：

- MC 样本均值作为点预测。
- MC 样本方差表示模型参数不确定性。
- 模型输出方差表示数据噪声不确定性。
- 总方差由二者合成，用于计算 90% 和 95% 预测区间。

## 6. 实验设置

默认实验配置位于 `configs/default.yaml`：

```yaml
seed: 42
output_dir: outputs

data:
  train_ratio: 0.7
  val_ratio: 0.15
  lookback: 32
  horizon: 16
  use_future_weather: false

model:
  name: improved_bnn
  hidden_dim: 256
  branch_dim: 64
  prior_sigma: 1.0

training:
  device: auto
  epochs: 80
  batch_size: 64
  lr: 0.00019544293665253097
  weight_decay: 0.0001
  kl_beta: 1.2114960587728605e-05
  patience: 12

prediction:
  mc_samples: 50

evaluation:
  run_test: true
```

时间序列任务不能使用随机划分。本项目按时间顺序切分数据：

```text
前 70%  -> 训练集
中 15%  -> 验证集
后 15%  -> 测试集
```

验证集用于 early stopping 和调参目标，测试集只用于最终性能评估。

## 7. 评价指标

项目同时评估点预测性能和概率预测质量。

### 7.1 点预测指标

| 指标 | 含义 | 趋势 |
| --- | --- | --- |
| `MAE` | 平均绝对误差 | 越小越好 |
| `RMSE` | 均方根误差，对大误差更敏感 | 越小越好 |
| `nRMSE` | 归一化 RMSE，便于不同量纲或容量比较 | 越小越好 |
| `sMAPE` | 对称平均绝对百分比误差 | 越小越好 |

### 7.2 概率预测指标

| 指标 | 含义 | 趋势 |
| --- | --- | --- |
| `NLL` | 高斯负对数似然，衡量预测分布拟合程度 | 越小越好 |
| `PICP` | Prediction Interval Coverage Probability，区间覆盖率 | 越接近目标置信度越好 |
| `PINAW` | Prediction Interval Normalized Average Width，归一化区间宽度 | 在覆盖率合理时越小越好 |

理想情况下，90% 区间的 `PICP` 应接近 `0.90`，95% 区间的 `PICP` 应接近 `0.95`。若 `PICP` 明显偏低，说明预测区间过窄；若 `PINAW` 过大，说明预测区间过宽。

## 8. 复现实验流程

### 8.1 安装依赖

```bash
pip install -r requirements.txt
```

如果需要使用 GPU，请安装与本机驱动匹配的 PyTorch CUDA 版本。例如 CUDA 12.8 轮子：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

检查 CUDA：

```bash
python -c "import numpy; import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 8.2 运行测试

```bash
pytest -q
```

当前测试覆盖数据处理、特征构造、窗口构造、数据泄漏检查、指标计算、模型输出维度、预测区间选择和调参辅助逻辑。

### 8.3 训练主模型

```bash
python -m src.train --config configs/default.yaml
```

训练日志中会输出设备信息：

```text
Device status: device=cuda, cuda_available=True, gpu=NVIDIA GeForce RTX ...
```

若显示 `device=cpu` 或 `cuda_available=False`，说明当前 Python 环境没有正确使用 CUDA。

### 8.4 自动调参

```bash
python -m src.tune
```

调参配置位于 `configs/tuning.yaml`。Optuna 默认搜索：

- `hidden_dim`
- `branch_dim`
- `lr`
- `kl_beta`

每个 trial 只优化验证集 RMSE，不评估测试集。调参结束后会导出：

```text
outputs/tuning/YYYYMMDD-HHMMSS/
├── best_params.json
├── trials.csv
└── best_config.yaml
```

### 8.5 模型对比

```bash
python -m src.compare_models
```

配置文件为 `configs/compare.yaml`。当前统一训练主流程已完整支持 `improved_bnn`；`src/models/baselines.py` 中已提供 MLP、CNN 和 MC Dropout 等 baseline 结构，后续可继续接入统一 trainer。

## 9. 输出结果说明

每次训练都会生成独立输出目录：

```text
outputs/improved_bnn/YYYYMMDD-HHMMSS/
```

主要文件如下：

```text
checkpoints/
├── best_model.pt
└── last_model.pt

metrics/
├── validation_metrics.json
├── metrics.json
├── point_metrics.csv
└── probabilistic_metrics.csv

figures/
├── loss_curve.png
├── prediction_interval_90.png
├── prediction_interval_95.png
├── horizon_rmse.png
├── picp_pinaw.png
└── calibration_curve.png

predictions/
├── test_predictions.csv
└── uncertainty_samples.npy

logs/
└── train.log

artifacts/
├── scaler_x.pkl
├── scaler_y.pkl
├── all_scalers.pkl
├── feature_columns.json
└── split_info.json
```

论文实验中建议重点使用：

- `metrics/metrics.json`：测试集总体指标。
- `metrics/point_metrics.csv`：逐预测步 MAE / RMSE。
- `figures/loss_curve.png`：训练集 `train_loss` 和验证集 `val_loss` 曲线。
- `figures/prediction_interval_90.png`：90% 预测区间图。
- `figures/prediction_interval_95.png`：95% 预测区间图。
- `figures/horizon_rmse.png`：预测步长误差曲线。
- `figures/picp_pinaw.png`：区间覆盖率和区间宽度对比。
- `figures/calibration_curve.png`：概率校准曲线。
- `predictions/test_predictions.csv`：测试集逐点预测值。

### 9.1 查看 train_loss 和 val_loss

每次训练时，程序会在终端和日志文件中逐轮输出：

```text
epoch=001 train_loss=... val_loss=...
```

训练结束后可以直接查看本次实验目录下的日志文件：

```text
outputs/improved_bnn/YYYYMMDD-HHMMSS/logs/train.log
```

例如在 PowerShell 中查看最近一次训练的 loss 记录：

```powershell
$log = Get-ChildItem outputs/improved_bnn -Recurse -Filter train.log |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Select-String -Path $log.FullName -Pattern "train_loss=.*val_loss="
```

如果只想看趋势图，打开对应实验目录中的：

```text
outputs/improved_bnn/YYYYMMDD-HHMMSS/figures/loss_curve.png
```

图中 `train` 曲线对应 `train_loss`，`val` 曲线对应 `val_loss`。一般来说，`train_loss` 反映模型对训练集的拟合情况，`val_loss` 反映模型在验证集上的泛化情况；本项目使用 `val_loss` 保存 `best_model.pt` 并触发 early stopping。

默认画图时会优先避开测试集开头的夜间零功率区间，从第一个真实出力大于 `prediction.plot.daylight_threshold` 的位置开始展示。如果需要固定展示某段时间，可以在配置中设置：

```yaml
prediction:
  plot:
    start_time: "2020-06-13 08:00:00"
    end_time: "2020-06-13 12:00:00"
```

## 10. 配置文件说明

项目包含三份主要配置：

| 文件 | 用途 |
| --- | --- |
| `configs/default.yaml` | 正式训练主配置 |
| `configs/tuning.yaml` | Optuna 调参配置 |
| `configs/compare.yaml` | 模型对比实验配置 |

关键参数说明：

| 参数 | 含义 |
| --- | --- |
| `seed` | 随机种子 |
| `output_dir` | 输出目录 |
| `data.generation_path` | 发电数据路径 |
| `data.weather_path` | 气象数据路径 |
| `data.fill_missing` | 是否补齐缺失时间点 |
| `data.train_ratio` | 训练集比例 |
| `data.val_ratio` | 验证集比例 |
| `data.lookback` | 历史窗口长度 |
| `data.horizon` | 预测窗口长度 |
| `data.use_future_weather` | 是否使用未来天气特征 |
| `model.hidden_dim` | 贝叶斯融合层和序列 MLP 隐藏维度 |
| `model.branch_dim` | 分支输出维度 |
| `model.prior_sigma` | 贝叶斯层权重先验标准差 |
| `training.device` | 训练设备，`auto` 优先使用 CUDA |
| `training.epochs` | 最大训练轮数 |
| `training.batch_size` | batch 大小 |
| `training.lr` | AdamW 学习率 |
| `training.weight_decay` | 权重衰减 |
| `training.kl_beta` | KL 散度项权重 |
| `training.patience` | early stopping 容忍轮数 |
| `prediction.mc_samples` | MC 前向传播次数 |
| `evaluation.run_test` | 是否评估测试集 |
| `tuning.n_trials` | Optuna trial 数 |

## 11. 代码结构

```text
.
├── configs/
│   ├── default.yaml
│   ├── tuning.yaml
│   └── compare.yaml
├── dataset/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
├── src/
│   ├── data.py              # CSV 读取、聚合与合并
│   ├── features.py          # 特征工程与特征分组
│   ├── dataset.py           # 时间切分、窗口构造和 Dataset
│   ├── losses.py            # Gaussian NLL 与 ELBO 损失
│   ├── metrics.py           # 点预测与概率预测指标
│   ├── evaluate.py          # 指标汇总
│   ├── predict.py           # MC 推理与区间计算
│   ├── visualization.py     # 实验图像绘制
│   ├── train.py             # 主训练入口
│   ├── tune.py              # Optuna 调参入口
│   ├── compare_models.py    # 模型对比入口
│   └── models/
│       ├── bayesian_layers.py
│       ├── branches.py
│       ├── improved_bnn.py
│       └── baselines.py
├── tests/
├── notebooks/
├── outputs/
├── requirements.txt
└── README.md
```

## 12. 常见问题

### 12.1 CUDA 不可用

先检查显卡驱动：

```bash
nvidia-smi
```

再检查 PyTorch：

```bash
python -c "import numpy; import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，通常是 PyTorch 安装版本与 CUDA 轮子不匹配。需要重新安装适合当前驱动环境的 PyTorch。

### 12.2 Windows / Anaconda OpenMP 报错

如果遇到：

```text
OMP: Error #15: Initializing libiomp5md.dll
```

通常是 Anaconda 环境中多个 OpenMP 运行时冲突。临时绕过方式：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.train --config configs/default.yaml
```

更推荐的方式是新建干净 Python 环境后重新安装依赖。

### 12.3 显存不足

RTX 3050 Laptop 这类 4GB 显存机器可优先调小：

```yaml
training:
  batch_size: 32

model:
  hidden_dim: 128
  branch_dim: 32

prediction:
  mc_samples: 20
```

### 12.4 训练太慢

可先使用小规模配置快速验证流程：

```yaml
training:
  epochs: 10
  batch_size: 128

prediction:
  mc_samples: 10

tuning:
  n_trials: 3
```

确认流程无误后，再恢复正式实验参数。

## 13. 论文写作参考

可在论文方法部分按如下逻辑描述：

1. 本文针对分布式光伏出力的超短期概率预测问题，构建过去 8 小时历史窗口到未来 4 小时预测窗口的监督学习样本。
2. 为避免时间序列数据泄漏，数据集按时间顺序划分，并且标准化器仅在训练集上拟合。
3. 模型采用多分支输入结构：气象特征输入全连接神经网络模块以学习复杂非线性关系，历史出力和历史气象序列输入一维卷积神经网络模块以提取局部时序模式，预测点前一时刻强相关变量通过直接输入分支引入贝叶斯概率融合层。
4. 融合层采用贝叶斯线性层，将权重建模为高斯变分后验，通过 KL 散度约束后验分布。
5. 模型输出预测均值和对数方差，使用 Gaussian NLL 与 KL 正则组成 ELBO 风格训练目标。
6. 推理阶段通过 Monte Carlo 多次前向传播估计模型不确定性，并结合输出方差得到预测区间。
7. 实验从点预测精度和概率预测质量两方面评估，指标包括 MAE、RMSE、nRMSE、sMAPE、NLL、PICP 和 PINAW。

可在实验局限性部分说明：当前公开数据集中的天气变量来自实测气象传感器，不等同于真实 NWP 预报。为避免未来实测气象造成信息泄漏，默认实验采用最近观测天气的持久化值作为预测窗口天气输入。后续若接入真实 NWP 数据，可进一步评估未来天气预报信息对光伏概率预测性能的提升。

## 附录：术语与英文缩写说明

术语表放在文档末尾，便于正文保持论文叙述的连贯性；阅读过程中遇到英文变量名、缩写或网络层名称时，可回到本节查询。

### 数据字段

| 名称 | 含义 | 在本文中的作用 |
| --- | --- | --- |
| `AC_POWER` | Alternating Current Power，交流输出功率 | 光伏逆变器输出到交流侧的功率，是本文默认预测目标 |
| `DC_POWER` | Direct Current Power，直流输入功率 | 光伏组件侧输入逆变器前的直流功率，是历史输入和直接输入特征 |
| `IRRADIATION` | Solar Irradiation，太阳辐照度 | 影响光伏出力的核心气象变量 |
| `AMBIENT_TEMPERATURE` | Ambient Temperature，环境温度 | 表示电站周围空气温度 |
| `MODULE_TEMPERATURE` | Module Temperature，组件温度 | 表示光伏组件表面温度，通常会影响发电效率 |
| `DATE_TIME` | 时间戳 | 用于时间排序、数据合并、周期特征构造和 train / val / test 切分 |

### 数据集与训练术语

| 名称 | 含义 |
| --- | --- |
| `train` | 训练集，用于更新模型参数 |
| `val` / validation | 验证集，用于 early stopping、调参和模型选择 |
| `test` | 测试集，只用于最终实验评估 |
| `lookback` | 历史输入窗口长度，例如 `32` 表示过去 32 个 15 分钟点 |
| `horizon` | 预测窗口长度，例如 `16` 表示未来 16 个 15 分钟点 |
| `batch` | 一次送入神经网络的一组样本 |
| `feature` | 输入特征变量，例如功率、辐照度、温度或时间编码 |
| `scaler` | 标准化器，用于把不同量纲的变量转换到更适合神经网络训练的尺度 |
| `NWP` | Numerical Weather Prediction，数值天气预报；当前数据集没有真实 NWP，因此默认不使用未来实测天气 |

### 模型与指标术语

| 名称 | 含义 |
| --- | --- |
| `BNN` | Bayesian Neural Network，贝叶斯神经网络 |
| `CNN` | Convolutional Neural Network，卷积神经网络 |
| `MLP` | Multi-Layer Perceptron，多层感知机 |
| `MC` | Monte Carlo，蒙特卡洛采样；本文用于多次前向传播估计不确定性 |
| `ELBO` | Evidence Lower Bound，变分贝叶斯常用目标；本文采用 Gaussian NLL + KL 正则的近似形式 |
| `KL` | Kullback-Leibler divergence，KL 散度，用于约束贝叶斯层后验分布不要过度偏离先验 |
| `NLL` | Negative Log Likelihood，负对数似然，衡量概率分布拟合质量 |
| `PICP` | Prediction Interval Coverage Probability，预测区间覆盖率 |
| `PINAW` | Prediction Interval Normalized Average Width，归一化预测区间宽度 |
| `mean` | 模型输出的预测均值，即点预测结果 |
| `log_var` | 模型输出的对数方差，用于表示预测分布的不确定性 |
