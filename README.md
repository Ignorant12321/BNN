# 基于深度学习的分布式光伏发电不确定性建模

本项目用于论文方向“基于深度学习的分布式光伏发电不确定性建模技术研究”。任务是基于 Plant 1 光伏电站历史发电数据与气象传感器数据，进行 4h 以内超短期光伏出力概率预测。

模型采用 PyTorch 实现，默认优先使用 GPU。主模型参考改进贝叶斯神经网络思想：历史功率序列使用 1D-CNN 分支提取时序波动，天气变量使用 MLP 分支提取非线性关系，时间特征和最近时刻特征单独输入，最后通过 BayesianLinear 概率层输出未来 16 个 15 分钟点的预测均值与不确定性。

## 项目结构

```text
.
├── dataset/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
├── configs/
│   ├── default.yaml
│   ├── tuning.yaml
│   └── compare.yaml
├── src/
│   ├── data.py
│   ├── features.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── tune.py
│   ├── compare_models.py
│   └── models/
│       ├── bayesian_layers.py
│       ├── branches.py
│       ├── improved_bnn.py
│       └── baselines.py
├── outputs/
├── tests/
├── notebooks/
├── requirements.txt
└── README.md
```

## 1. 准备数据

把两个原始 CSV 放到 `dataset/` 目录：

```text
dataset/Plant_1_Generation_Data.csv
dataset/Plant_1_Weather_Sensor_Data.csv
```

代码会自动完成：

- 解析 `DATE_TIME`
- 将 22 个逆变器数据聚合为电站级数据
- 按时间合并气象数据
- 补齐 15 分钟时间序列
- 构造 8h 历史窗口到未来 4h 预测窗口
- 默认使用最近实测气象的持久化值作为未来天气输入，避免把未来实测气象泄漏给模型

默认预测目标是电站级 `AC_POWER`。

## 2. 安装环境

建议在项目根目录执行：

```bash
pip install -r requirements.txt
```

然后安装 PyTorch GPU 版本。当前机器如果使用 CUDA 12.8 轮子，可执行：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

如果你的 CUDA / 驱动环境不同，可以到 PyTorch 官网选择对应安装命令。

检查 GPU 是否可用：

```bash
python -c "import numpy; import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

输出中如果看到 `True` 和显卡名称，说明 PyTorch 可以使用 GPU。

## 3. 运行测试

先运行单元测试，确认代码和数据处理逻辑正常：

```bash
pytest -q
```

正常情况下应看到类似：

```text
11 passed
```

测试覆盖内容包括：

- CSV 读取与时间解析
- 逆变器聚合
- 时间特征构造
- 滑动窗口构造
- 数据泄漏检查
- 指标计算
- BNN 模型输出维度
- Gaussian NLL 损失反向传播

## 4. 训练主模型

默认训练命令：

```bash
python -m src.train --config configs/default.yaml
```

默认配置位于 [configs/default.yaml](configs/default.yaml)。

关键参数：

```yaml
data:
  lookback: 32    # 过去 32 个 15 分钟点，即 8h
  horizon: 16     # 未来 16 个 15 分钟点，即 4h
  use_future_weather: false  # 无真实 NWP 时保持 false，避免未来气象泄漏

training:
  device: auto    # 自动使用 cuda，无法使用时回退 cpu
  epochs: 80
  batch_size: 64
  lr: 0.001

prediction:
  mc_samples: 50  # MC 前向传播次数，用于估计不确定性
  plot:
    prefer_daylight: true      # 默认避开夜间零功率段
    daylight_threshold: 1.0
    max_points: 160
    # start_time: "2020-06-13 08:00:00"
    # end_time: "2020-06-13 12:00:00"

evaluation:
  run_test: true  # 正式训练评估测试集；Optuna trial 会自动改为 false
```

训练时日志会显示设备：

```text
Device status: device=cuda, cuda_available=True, gpu=NVIDIA GeForce RTX ...
```

如果 `device=cpu` 或 `cuda_available=False`，说明当前 Python 环境没有让 PyTorch 使用 CUDA。

## 5. 查看输出结果

每次训练会自动生成一个新的结果目录：

```text
outputs/improved_bnn/YYYYMMDD-HHMMSS/
```

例如：

```text
outputs/improved_bnn/20260509-211443/
```

目录内容：

```text
checkpoints/
├── best_model.pt
└── last_model.pt

metrics/
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

重点看这些文件：

- `metrics/metrics.json`：总体 MAE、RMSE、nRMSE、sMAPE、NLL、PICP、PINAW
- `figures/prediction_interval_90.png`：90% 预测区间
- `figures/prediction_interval_95.png`：95% 预测区间
- `figures/horizon_rmse.png`：未来 1 到 16 步误差变化
- `figures/calibration_curve.png`：概率预测校准效果
- `predictions/test_predictions.csv`：测试集逐点预测结果
- `predictions/uncertainty_samples.npy`：MC 采样结果

默认情况下，预测区间图会自动跳过测试集开头的夜间零功率点，从第一个真实出力大于 `prediction.plot.daylight_threshold` 的时刻开始展示。如果希望固定展示某个时间段，可在配置文件中填写：

```yaml
prediction:
  plot:
    start_time: "2020-06-13 08:00:00"
    end_time: "2020-06-13 12:00:00"
```

指定时间段会优先于自动白天选择。

## 6. 指标含义

点预测指标：

- `MAE`：平均绝对误差，越小越好
- `RMSE`：均方根误差，越小越好
- `nRMSE`：归一化 RMSE，便于不同容量场景比较
- `sMAPE`：对称平均百分比误差，越小越好

概率预测指标：

- `NLL`：负对数似然，越小表示概率分布拟合越好
- `PICP`：预测区间覆盖率，越接近目标置信度越好
- `PINAW`：归一化区间宽度，越小表示区间越窄

理想情况下，90% 区间的 PICP 应接近 `0.90`，95% 区间的 PICP 应接近 `0.95`。如果 PICP 很低，说明区间过窄；如果 PINAW 很大，说明区间过宽。

## 7. 调参

运行 Optuna 调参：

```bash
python -m src.tune
```

调参配置在 [configs/tuning.yaml](configs/tuning.yaml)。Optuna trial 只保存并优化验证集 RMSE，不评估测试集；测试集只用于最终正式实验。默认会搜索：

- `hidden_dim`
- `branch_dim`
- `lr`
- `kl_beta`

建议先用较少的 `n_trials` 快速试跑，确认流程无误后再增加次数。

## 8. 模型对比

对比实验入口：

```bash
python -m src.compare_models
```

配置文件是 [configs/compare.yaml](configs/compare.yaml)。当前主流程已经完整支持 `improved_bnn`；基线模型结构已放在 `src/models/baselines.py`，后续可以继续扩展统一训练逻辑。

论文里建议至少比较：

- Improved BNN
- MLP baseline
- CNN baseline
- MC Dropout

## 9. 常见问题

### 9.1 CUDA 不可用

先检查：

```bash
nvidia-smi
```

再检查 PyTorch：

```bash
python -c "import numpy; import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，通常是 PyTorch 版本和 CUDA 轮子不匹配。重新安装对应 CUDA 版本的 PyTorch。

### 9.2 Windows / Anaconda 出现 OpenMP 报错

如果遇到类似：

```text
OMP: Error #15: Initializing libiomp5md.dll
```

当前环境中可先用以下方式检查：

```bash
python -c "import numpy; import torch; print(torch.cuda.is_available())"
```

如果这样可以正常运行，说明是 Anaconda 环境中的 OpenMP 加载顺序冲突。项目训练入口通常会先加载 numpy/pandas，再加载 torch，因此可以正常训练。

临时绕过方式：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.train --config configs/default.yaml
```

更推荐的长期方案是新建干净环境再安装 PyTorch。

### 9.3 显存不足

RTX 3050 Laptop 这类 4GB 显存机器建议先调小：

```yaml
training:
  batch_size: 32

model:
  hidden_dim: 64
  branch_dim: 32

prediction:
  mc_samples: 20
```

### 9.4 训练太慢

可以先做快速实验：

```yaml
training:
  epochs: 10
  batch_size: 128

prediction:
  mc_samples: 10
```

等流程稳定后，再恢复正式参数。

## 10. 论文写作提示

本文实验可以这样描述：

- 数据来源：Plant 1 光伏发电与气象传感器数据
- 时间粒度：15 分钟
- 预测任务：未来 4h，即 16 步超短期预测
- 输入窗口：过去 8h，即 32 步历史数据
- 模型结构：历史 CNN 分支、天气 MLP 分支、时间分支、直接输入分支、贝叶斯概率层
- 不确定性估计：通过 BayesianLinear 权重采样和模型输出方差共同构造预测分布
- 评价指标：MAE、RMSE、nRMSE、sMAPE、NLL、PICP、PINAW

需要注意：当前公开数据集中使用的是实测气象数据，不是真正的数值天气预报 NWP。默认实验使用最近时刻气象持久化作为未来天气输入，避免使用预测窗口内的未来实测气象；后续若接入真实 NWP，可将 `use_future_weather` 改为 `true` 或替换为 NWP 特征来源。
