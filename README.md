# 光伏功率递归概率预测实验

本项目用于复现 15 分钟粒度光伏功率 4 小时超短期预测实验。当前论文主线方法是 `pv_usibnn_recursive_4h`：使用标准 Recursive 策略训练单步 Bayesian IBNN，再递推生成未来 16 个时间步预测。

核心实验包括：

```text
1. 本文方法训练
   pv_usibnn_recursive_4h

2. 点预测对比
   改进 BNN vs MLP vs 1D-CNN vs LSTM

3. 区间预测对比
   本文 BNN 概率区间 vs 正态残差区间法 vs 前一天同时间持续区间法

4. Optuna 调参
   针对 pv_usibnn_recursive_4h 搜索 lr 和 kl_beta
```

## 环境准备

安装依赖：

```powershell
pip install -r requirements.txt
```

检查 PyTorch / CUDA：

```powershell
python -m src.environment
```

`training.device: auto` 会优先使用 CUDA；没有 CUDA 时使用 CPU。

## 数据预处理

原始数据默认位置：

```text
dataset/Plant_1_Generation_Data.csv
dataset/Plant_1_Weather_Sensor_Data.csv
```

先预处理：

```powershell
python -m src.data.preprocess --config configs/data.yaml
```

再按时间顺序切分：

```powershell
python -m src.data.split --config configs/data.yaml
```

输出：

```text
data/processed/plant_frame.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

训练入口要求 `train/val/test` split 已存在。如果原始 CSV 存在但 split 缺失，程序会提示先运行上述两个命令。

## 本文方法

主配置：

```text
configs/models/bnn/pv_usibnn_recursive_4h.yaml
```

任务设置：

```text
lookback = 16    # 过去 4 小时，15min x 16
horizon  = 16    # 未来 4 小时，15min x 16
strategy = recursive
```

输入特征：

```text
history:
  AC_POWER

weather:
  IRRADIATION
  AMBIENT_TEMPERATURE
  MODULE_TEMPERATURE
  hour_sin
  hour_cos
  dayofyear_sin
  dayofyear_cos
  is_generation_time

direct:
  AC_POWER
```

模型结构：

```text
历史功率 -> Bayesian 1D-CNN 分支
未来气象/时间特征 -> Bayesian FC 分支
前一时刻功率/上一时刻预测 -> direct 输入融合
融合层 -> mean + log_var
MC sampling -> 概率预测区间
```

训练损失：

```text
Loss = Gaussian NLL + kl_beta * KL(q(w)||p(w))
```

代码中 Gaussian NLL 省略了与参数无关的常数项 `ln(2π)`；`kl_beta` 等价于论文公式中 KL 权重和 mini-batch 缩放的合并系数。

## 训练本文方法

推荐命令：

```powershell
python -m src.experiments.train --config configs/models/bnn/pv_usibnn_recursive_4h.yaml
```

该命令会自动识别：

```yaml
strategy:
  name: recursive
```

因此实际流程是：

```text
4h 配置 -> 切成 horizon=1 单步模型 -> 训练单步模型 -> 外部递归滚动 16 步
```

也可以显式使用递归训练入口：

```powershell
python -m src.experiments.train_bnn_recursive_4h --config configs/models/bnn/pv_usibnn_recursive_4h.yaml
```

训练输出目录：

```text
outputs/train/pv_usibnn_recursive/<训练时间戳>/
```

主要文件：

```text
config.yaml
manifest.json
train.log
epoch_history.csv
metrics.csv
predictions/test.csv
figures/loss_curve.png
figures/prediction_0800_1200.png
figures/prediction_1000_1400.png
figures/prediction_1200_1600.png
figures/prediction_window_metrics.csv
models/best.pt
```

`metrics.csv` 包含全时段和有效发电时段指标：

```text
test_mae
test_rmse
test_nmae
test_nrmse
test_picp_90
test_pinaw_90
test_picp_95
test_pinaw_95
test_generation_*
```

有效发电时段定义为：

```text
06:00 <= target_time <= 18:00
```

## 点预测对比

运行：

```powershell
python -m src.experiments.compare_recursive_point_forecasts_4h
```

默认对比：

```text
BNN    -> configs/models/bnn/pv_usibnn_recursive_4h.yaml
MLP    -> configs/models/mlp/recursive_4h.yaml
1D-CNN -> configs/models/cnn/recursive_4h.yaml
LSTM   -> configs/models/lstm/recursive_4h.yaml
```

上述点预测 baseline 与 `pv_usibnn_recursive_4h` 使用相同的历史功率、未来气象/时间特征和 direct 功率输入；MLP、1D-CNN、LSTM 均采用 32 维小容量配置，避免由 64/128 隐藏维度带来的容量差异。

点预测对比只输出点预测指标，不比较区间指标：

```text
MAE
RMSE
NMAE
NRMSE
generation MAE/RMSE/NMAE/NRMSE
```

点预测窗口图只画真实值和各模型点预测均值，不画 90%/95% 预测区间；区间图和区间指标请使用后面的区间对比脚本。

输出目录：

```text
outputs/comparisons/recursive_point_forecasts_4h_<时间戳>/
```

主要文件：

```text
model_metrics.csv
summary.md
predictions/BNN.csv
predictions/MLP.csv
predictions/1D-CNN.csv
predictions/LSTM.csv
figures/loss_curves.png
figures/metrics_test_mae.png
figures/metrics_test_nrmse.png
figures/prediction_0800_1200.png
figures/prediction_1000_1400.png
figures/prediction_1200_1600.png
```

图中的纵轴单位为：

```text
AC Power (kW)
```

如果想临时降低训练量：

```powershell
python -m src.experiments.compare_recursive_point_forecasts_4h --epochs 50 --n-samples 10
```

## 点预测 baseline 说明

点预测 baseline 保持普通模型结构，用来证明本文方法相对经典模型的优势：

```text
MLP:
  历史功率 + 未来气象/时间特征 + direct -> 展平
  32/32 全连接网络

1D-CNN:
  历史功率序列 -> 普通 1D-CNN
  未来气象/时间特征 + direct -> 展平
  拼接后 FC 融合

LSTM:
  历史功率序列 -> LSTM
  未来气象/时间特征 + direct -> 展平
  拼接后 FC 融合

改进 BNN:
  历史功率 -> Bayesian 1D-CNN
  未来气象/时间特征 -> Bayesian FC
  direct -> 融合
  输出 mean/log_var，并用 MC sampling 得到概率区间
```

注意：普通 MLP、1D-CNN、LSTM 是确定性点预测模型，训练损失为 MSE；本文 BNN 是概率模型，训练损失为 Gaussian NLL + KL。

## 区间预测对比

先训练本文方法：

```powershell
python -m src.experiments.train --config configs/models/bnn/pv_usibnn_recursive_4h.yaml
```

然后把 `<训练时间戳>` 换成实际目录名：

```powershell
python -m src.experiments.compare_recursive_interval_methods_4h --run outputs/train/pv_usibnn_recursive/<训练时间戳>
```

输出目录：

```text
outputs/comparisons/recursive_interval_methods_4h_<时间戳>/
```

主要文件：

```text
coverage_summary.csv
calibrated_coverage_summary.csv
predictions/our_method.csv
predictions/normal_distribution.csv
predictions/persistence_interval.csv
```

对比方法：

```text
our_method:
  本文 BNN 自身输出的 mean/log_var 区间。

normal_distribution:
  基于本文模型点预测的验证集残差正态区间。
  对每个 horizon 估计 residual_mean 和 residual_std。

persistence_interval:
  前一天同时间持续区间。
  优先使用 target_time - 1 day 的真实功率作为持续预测中心；
  若找不到前一天同时间样本，则回退到 direct AC_POWER。
```

`coverage_summary.csv` 输出：

```text
confidence
our_method_picp
our_method_pinaw
normal_picp
normal_pinaw
persistence_picp
persistence_pinaw
```

`calibrated_coverage_summary.csv` 会在验证集上按目标 PICP 校准区间宽度，再报告测试集 PICP/PINAW。

## Optuna 调参

调参配置：

```text
configs/tune/pv_usibnn_recursive_4h.yaml
```

运行：

```powershell
python -m src.experiments.tune --config configs/tune/pv_usibnn_recursive_4h.yaml --note "PV USIBNN Recursive 4h tuning"
```

默认搜索：

```text
training.lr
training.kl_beta
```

目标指标：

```text
val_generation_nrmse
```

输出目录：

```text
outputs/tuning/pv_usibnn_recursive_4h_generation_optuna/
```

主要文件：

```text
tuning_config.yaml
study.db
trials.csv
best_config.yaml
best_run.txt
runs/trial-0000/
```

预览最优参数写回：

```powershell
python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/pv_usibnn_recursive_4h_generation_optuna --target configs/models/bnn/pv_usibnn_recursive_4h.yaml
```

确认直接写回：

```powershell
python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/pv_usibnn_recursive_4h_generation_optuna --target configs/models/bnn/pv_usibnn_recursive_4h.yaml --yes
```

## 其他实验入口

递归策略、Direct、MIMO BNN 策略对比：

```powershell
python -m src.experiments.compare_bnn_strategies_4h
```

本文方法输入分支消融：

```powershell
python -m src.experiments.ablate_bnn_recursive_4h --config configs/models/bnn/pv_usibnn_recursive_4h.yaml
```

已训练模型通用对比：

```powershell
python -m src.experiments.compare --config configs/compare/main.yaml
```

本地可视化：

```powershell
python visualizer/server.py
```

然后打开终端显示的地址，通常是：

```text
http://127.0.0.1:5177/
```

## 完整推荐流程

```powershell
python -m src.environment

python -m src.data.preprocess --config configs/data.yaml
python -m src.data.split --config configs/data.yaml

python -m src.experiments.tune --config configs/tune/pv_usibnn_recursive_4h.yaml --note "PV USIBNN Recursive 4h tuning"
python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/pv_usibnn_recursive_4h_generation_optuna --target configs/models/bnn/pv_usibnn_recursive_4h.yaml --yes

python -m src.experiments.train --config configs/models/bnn/pv_usibnn_recursive_4h.yaml

python -m src.experiments.compare_recursive_point_forecasts_4h
python -m src.experiments.compare_recursive_interval_methods_4h --run outputs/train/pv_usibnn_recursive/<训练时间戳>
```

## 测试

运行全部测试：

```powershell
pytest -q
```

运行和当前主线相关的测试：

```powershell
pytest tests/test_config.py::test_pv_usibnn_recursive_four_hour_config_uses_standard_recursive_strategy tests/test_bnn_recursive_4h_experiment.py tests/test_recursive_point_forecast_comparison.py tests/test_recursive_interval_comparison.py tests/test_plots.py -q
```
