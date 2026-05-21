# 光伏功率概率预测实验框架

本项目用于做光伏功率预测实验。当前流程分为三步：先把原始 CSV 预处理并按时间切分，再按单个 YAML 配置训练一个模型，最后加载已训练产物做统一评估和对比。

训练阶段只使用 `train` 拟合模型，并输出 `train/val` 指标；`test` 指标只在 `compare` 阶段统一计算。这样可以避免在调参时反复使用测试集。Torch 后端会只用 train split 拟合标准化参数，在标准化空间训练模型，并在指标、预测 CSV 和图表中反变换回原始 `AC_POWER` 量纲。

## 环境

```powershell
pip install -r requirements.txt
```

可选：检查 PyTorch 和 CUDA 状态。

```powershell
python -m src.environment
```

项目默认训练后端是 `torch`，设备是 `auto`。`auto` 会优先使用 CUDA，没有可用 CUDA 时使用 CPU。也可以在模型配置里固定设备：

```yaml
training:
  device: cpu
```

## 目录结构

```text
configs/
  data.yaml                 # 原始数据路径、切分比例、窗口长度、特征列
  models/
    bnn/
      bnn.yaml              # improved_bnn 默认配置
      1h.yaml               # 只覆盖 lookback=4
      4h.yaml               # 只覆盖 lookback=16
      8h.yaml               # 只覆盖 lookback=32
      12h.yaml              # 只覆盖 lookback=48
      24h.yaml              # 使用默认 lookback=96
    mlp/
      mlp.yaml              # mlp_baseline 默认配置
      24h.yaml
    cnn/
      cnn.yaml              # cnn_baseline 默认配置
      24h.yaml
    mc_dropout/
      mc_dropout.yaml       # mc_dropout 默认配置
      24h.yaml
  compare/
    main.yaml               # 主模型和 baseline 对比
    lookback.yaml           # improved_bnn 不同 lookback 对比
  tune/
    bnn.yaml                # Optuna 调参配置
src/
  data/                     # 预处理、切分、滑动窗口构造
  models/                   # 模型注册表、PyTorch 模型、NumPy fallback
  training/                 # 训练流程和 PyTorch 训练器
  evaluation/               # 指标、预测导出、图表、统一评估
  artifacts/                # 训练/对比产物读写
  experiments/
    train.py                # 单模型训练入口
    tune.py                 # Optuna 调参入口
    apply_tuning.py         # 预览并应用调参最优参数
    compare.py              # 已训练产物对比入口
```

## 数据流程

原始数据默认放在 `dataset/`：

```text
dataset/Plant_1_Generation_Data.csv
dataset/Plant_1_Weather_Sensor_Data.csv
```

预处理会按 `DATE_TIME` 聚合发电数据，聚合方式是 `DC_POWER/AC_POWER/DAILY_YIELD/TOTAL_YIELD` 求和；天气数据按同一时间戳求均值；两者 inner join 后写出：

```powershell
python -m src.data.preprocess --config configs/data.yaml
```

输出：

```text
data/processed/plant_frame.csv
```

然后按时间顺序切分，不随机打乱：

```powershell
python -m src.data.split --config configs/data.yaml
```

输出：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

训练入口要求这些 split 文件已经存在。如果原始 CSV 存在但 split 缺失，程序会报错并提示先运行预处理和切分命令。

## 窗口和特征

`configs/data.yaml` 中的默认任务是 15 分钟粒度下，用过去 24 小时预测未来 4 小时：

```yaml
data:
  lookback: 96
  horizon: 16
  features:
    history:
      - AC_POWER
    weather:
      - AMBIENT_TEMPERATURE
      - MODULE_TEMPERATURE
      - IRRADIATION
    direct:
      - AC_POWER
    target: AC_POWER
```

每个 split 会独立构造滑动窗口，不会让验证集或测试集开头的窗口借用前一个 split 末尾的历史行。单个窗口含义如下：

```text
history: 过去 lookback 步的历史特征
weather: 未来 horizon 步的天气特征
direct : 预测起点前一时刻的直接特征
target : 未来 horizon 步的 AC_POWER
```

常用 lookback 配置：

```text
1h  = 4
4h  = 16
8h  = 32
12h = 48
24h = 96
```

## 训练

每次训练一个模型：

```powershell
python -m src.experiments.train --config configs/models/bnn/24h.yaml
```

训练不同 lookback 的 improved_bnn：

```powershell
python -m src.experiments.train --config configs/models/bnn/1h.yaml
python -m src.experiments.train --config configs/models/bnn/4h.yaml
python -m src.experiments.train --config configs/models/bnn/8h.yaml
python -m src.experiments.train --config configs/models/bnn/12h.yaml
python -m src.experiments.train --config configs/models/bnn/24h.yaml
```

训练 24h baseline：

```powershell
python -m src.experiments.train --config configs/models/mlp/24h.yaml
python -m src.experiments.train --config configs/models/cnn/24h.yaml
python -m src.experiments.train --config configs/models/mc_dropout/24h.yaml
```

训练产物目录：

```text
outputs/train/<model_name>/<timestamp>/
  config.yaml
  manifest.json
  train.log
  epoch_history.csv
  metrics.csv
  figures/loss_curve.png
  models/best.pt             # torch 后端，按验证集 RMSE 回滚后的最佳 epoch
  models/best.pkl            # numpy 后端
```

`metrics.csv` 只包含 `train` 和 `val` 指标。指标包括：

```text
mae, rmse, nmae, nrmse, picp_90, pinaw_90, picp_95, pinaw_95
```

## 模型配置

模型通过 `model.name` 从注册表构造。当前支持：

```text
improved_bnn
mlp_baseline
cnn_baseline
mc_dropout
```

当前 `configs/models/*/*.yaml` 都使用 torch 后端。四个模型在 torch 后端是独立结构，不再共用同一个网络：

```text
improved_bnn : 改进 BNN，包含 BayesianLinear、BayesianConv1d 和 KL loss
mlp_baseline : 普通 MLP，使用 history + future weather + direct
cnn_baseline : 真正 Conv1D baseline，只使用 history 序列
mc_dropout   : MC Dropout，使用 history + future weather，预测阶段多次 dropout sampling
```

`improved_bnn` 的 torch 结构按表 3 固定，不再用 `hidden_dim`、`branch_dim` 或 `conv_kernel` 调整网络层数/单元数：

```text
第一部分 history 展平输入
  -> 概率全连接层 32
  -> 概率全连接层 64
  -> 概率全连接层 16

第二部分 history 序列输入
  -> 1D 概率卷积层 32，核 5
  -> 平均池化层，核 5
  -> 1D 概率卷积层 32，核 5
  -> 全局平均池化层

第三部分 future weather + direct 输入
  -> 作为第三输入分支直接参与合并

合并层
  -> 概率全连接层 32
  -> 概率全连接层 16
  -> 输出层 mean / log_var
```

如果把 `training.backend` 改成 `numpy`，注册表会构造轻量 ridge regression 版本，用于测试或无 PyTorch 场景。

常调字段：

```yaml
model:
  hidden_dim: 128    # mlp_baseline、cnn_baseline、mc_dropout 使用
  branch_dim: 64     # cnn_baseline、mc_dropout 使用
  dropout: 0.2       # 仅 mc_dropout 使用

training:
  epochs: 50
  batch_size: 64
  lr: 0.0005
  weight_decay: 0.0001
  kl_beta: 0.000001  # improved_bnn 使用

evaluation:
  n_samples: 30      # improved_bnn 和 mc_dropout 训练期验证/对比评估的预测采样次数
```

Torch 模型默认启用标准化；保存的 `config.yaml` 会包含本次 run 的 scaler，因此 `compare` 会先用同一 scaler 转换输入，再把预测均值、方差和 target 恢复到原始量纲后计算指标。若确实需要关闭，可在配置中写：

```yaml
data:
  scaling:
    enabled: false
```

## Optuna 调参

当前提供 BNN 的 Optuna 调参入口：

```powershell
python -m src.experiments.tune --config configs/tune/bnn.yaml
```

默认配置在 `configs/tune/bnn.yaml`，会基于 `configs/models/bnn/24h.yaml` 搜索超参数，并用验证集 `val_rmse` 作为目标指标。默认搜索：

```text
lr, weight_decay, kl_beta, batch_size
```

BNN 的网络结构当前按表 3 固定，因此 Optuna 只搜索训练超参数，不再搜索 `hidden_dim`、`branch_dim` 或 `conv_kernel`。

Optuna 只负责生成超参数组合；每个 trial 仍然调用项目现有训练流程。只要训练配置是 `training.backend: torch`，训练器就会使用 `torch.optim.AdamW`，因此 BNN 调参时实际是：

```text
Optuna 选参数 -> AdamW 训练 -> 读取 val_rmse -> Optuna 更新搜索
```

调参输出是稳定目录，不带时间戳：

```text
outputs/tuning/bnn_optuna/
  tuning_config.yaml
  study.db
  trials.csv
  best_config.yaml
  best_run.txt
  runs/trial-0000/
  runs/trial-0001/
```

`study.db` 是 Optuna 的 SQLite storage，用于断点继续。`n_trials` 表示目标完成 trial 总数：如果中断后只完成了 7/20，再运行同一个命令会继续补到 20；如果已经完成 20/20，再运行不会重复训练。需要继续扩展搜索时，把 `configs/tune/bnn.yaml` 里的 `n_trials` 改大即可。

调参 trial 的训练产物不会写到顶层 `outputs/train/`，而是写入当前 tuning 目录下的 `runs/trial-xxxx/`。

应用最优参数前可以先预览变化：

```powershell
python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/bnn_optuna --target configs/models/bnn/bnn.yaml
```

命令会展示类似：

```text
Tuned Parameter Changes
------------------------------------------------
training.lr      | 0.0005 -> 0.0009291350888559107
training.kl_beta | 1e-06 -> 4.182709268632557e-05
training.batch_size | 64 -> 128
```

然后提示是否应用。确认后才会写回目标 YAML。若确认无误，也可以跳过提示：

```powershell
python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/bnn_optuna --target configs/models/bnn/bnn.yaml --yes
```

## 对比评估

对比命令不会重新训练模型。它会加载训练目录中的 best 模型，在指定 split 上重新预测并计算指标，默认 split 是 `test`。

使用 YAML：

```powershell
python -m src.experiments.compare --config configs/compare/main.yaml
python -m src.experiments.compare --config configs/compare/lookback.yaml
```

也可以直接从命令行传入训练产物：

```powershell
python -m src.experiments.compare --run BNN-24h=outputs/train/improved_bnn/20260521-105753
python -m src.experiments.compare --name lookback --run outputs/train/improved_bnn
python -m src.experiments.compare --name main --run BNN-24h=outputs/train/improved_bnn/20260521-105753 --run outputs/train/mlp_baseline
```

当路径指向单个训练目录，例如 `outputs/train/improved_bnn/20260521-105753`，只评估这一份训练产物。写成 `label=path` 时会使用 `label` 作为显示名。

当路径指向模型根目录，例如 `outputs/train/improved_bnn`，程序会自动展开其下所有训练目录逐个对比，默认 label 使用每个训练目录名。因此你可以把时间戳目录改名为 `1h`、`4h`、`best-24h` 这类更可读的名字，再直接对比整个模型根目录：

```yaml
runs:
  - path: outputs/train/improved_bnn
```

如果需要自定义 label，请在 YAML 中逐个列出具体训练目录：

```yaml
runs:
  - label: BNN-24h
    path: outputs/train/improved_bnn/best-24h
```

对比产物目录：

```text
outputs/comparisons/<timestamp>/
  compare_config.yaml
  model_metrics.csv
  predictions/<label>.csv
  figures/metrics_<metric>.png
  figures/loss_curves.png
  figures/prediction_0800_1200.png
  figures/prediction_1000_1400.png
  figures/prediction_1200_1600.png
  report.md
```

旧入口 `src.experiments.compare_results` 已删除，请使用 `src.experiments.compare`。

## 配置 include

模型配置通常 include 公共数据配置：

```yaml
include:
  - bnn.yaml

data:
  lookback: 48
```

include 文件先加载，当前 YAML 中的同名字段会覆盖 include 中的值。模型族默认配置放在同目录的 `<model>.yaml` 中，例如 `configs/models/bnn/bnn.yaml`；具体实验文件只写差异字段。

## 建议实验顺序

1. 先运行预处理和切分。
2. 先用 `configs/tune/bnn.yaml` 在 24h BNN 上搜索一组候选超参数。
3. 用 `outputs/tuning/bnn_optuna/best_config.yaml` 重训或作为参考更新 BNN 默认配置。
4. 固定候选超参数后，训练 `improved_bnn` 和三个 baseline。
5. 用 `configs/compare/main.yaml` 做 24h 主模型和 baseline 对比。
6. 分别训练 `1h/4h/8h/12h/24h` 的 `improved_bnn`。
7. 用 `configs/compare/lookback.yaml` 做不同历史窗口对比。
8. 只用 `train/val` 调参，最后再用 `test` 结果写最终对比。

## 测试

```powershell
pytest -q
```
