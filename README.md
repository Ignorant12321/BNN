# 光伏概率预测实验框架

这个项目用于做光伏功率概率预测实验。当前工程流程被拆成三步：

1. 数据预处理与切分；
2. 每次只训练一个模型，并只用 `train.csv` 拟合；
3. 单独读取已有训练结果做对比。

这样做的好处是：不同模型、不同 lookback、不同参数都能形成独立 run，后续对比不会重新训练，也更适合论文实验管理。

## 目录结构

```text
configs/
  data.yaml                 # 数据路径、窗口长度、特征列、切分比例
  models/
    bnn_1h.yaml             # BNN，过去 1h 输入
    bnn_4h.yaml             # BNN，过去 4h 输入
    bnn_8h.yaml             # BNN，过去 8h 输入
    bnn_24h.yaml            # BNN，过去 24h 输入
    mlp_24h.yaml            # MLP baseline，过去 24h 输入
    cnn_24h.yaml            # CNN baseline，过去 24h 输入
    mc_dropout_24h.yaml     # MC Dropout baseline，过去 24h 输入
  compare/
    main.yaml               # 读取已有 run，对比主模型和 baseline
    lookback.yaml           # 读取已有 run，对比 BNN 不同 lookback
src/
  config.py                 # YAML include、默认配置和配置合并
  environment.py            # PyTorch/CUDA 环境检测
  data/
    pv.py                   # 数据读取、合并、按 split 构造滑动窗口
    preprocess.py           # 原始 CSV 预处理命令
    split.py                # train/val/test 切分命令
  models/
    registry.py             # 模型注册表
    torch_models.py          # PyTorch/CUDA 模型
    improved_bnn.py         # NumPy 轻量主模型
    baselines.py            # NumPy ridge baseline
  training/
    trainer.py             # 通用训练编排：fit、train/val 指标、best epoch
    torch_trainer.py         # PyTorch/CUDA 训练器
  evaluation/
    predictor.py           # 统一预测输出
    evaluator.py           # 加载 run 并统一评估
    metrics.py             # 统一指标
    plots.py               # 轻量对比图
  artifacts/
    run_io.py              # run/comparison 目录、配置、模型读写
    manifest.py            # run 元数据
  experiments/
    train.py                # 单模型训练入口
    compare.py              # 加载已训练模型，统一评估并对比
    compare_results.py      # 兼容旧入口，转到 compare.py
```

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 检查 PyTorch/CUDA

```powershell
python -m src.environment
```

正常情况下会看到类似：

```text
available: True
cuda_available: True
device: cuda
cuda_device_name: NVIDIA ...
```

项目默认训练后端是 PyTorch，默认设备是 CUDA。这些默认值由 `src/config.py` 补齐，所以模型 yaml 里不用重复写：

```yaml
training:
  backend: torch
  device: cuda
```

如果需要临时改成 CPU，可以在某个模型 yaml 里覆盖：

```yaml
training:
  device: cpu
```

## 数据配置

数据默认配置在：

```text
configs/data.yaml
```

关键字段：

```yaml
data:
  lookback: 96     # 默认过去 24h，15 分钟粒度下 96 步
  horizon: 16      # 预测未来 4h，15 分钟粒度下 16 步
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

`history_features / weather_features / direct_features` 不需要手写，代码会根据 `data.features` 自动推断。

## 数据预处理

```powershell
python -m src.data.preprocess --config configs/data.yaml
```

输出：

```text
data/processed/plant_frame.csv
```

## 划分训练集、验证集、测试集

```powershell
python -m src.data.split --config configs/data.yaml
```

输出：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

切分按时间顺序完成，不会随机打乱。训练入口会直接读取这三个文件；如果原始 CSV 存在但缺少任意 split 文件，训练会报错并提示先运行预处理和切分命令。

## 训练单个模型

每个 `configs/models/*.yaml` 都是一份单模型训练配置。训练前请先完成：

```powershell
python -m src.data.preprocess --config configs/data.yaml
python -m src.data.split --config configs/data.yaml
```

项目当前没有 `configs/default.yaml`。单模型训练命令应使用 `configs/models/` 下的具体模型配置。

训练时数据流如下：

```text
data/processed/train.csv -> 构造 train 窗口 -> 拟合模型
data/processed/val.csv   -> 构造 val 窗口   -> 训练过程评估/选择 best
data/processed/test.csv  -> 由 compare 统一读取和评估
```

为了避免数据泄露，`train`、`val`、`test` 会在各自 CSV 内独立构造滑动窗口。也就是说，验证集或测试集开头的窗口不会借用训练集末尾的历史行。

训练 BNN 24h：

```powershell
python -m src.experiments.train --config configs/models/bnn_24h.yaml
```

训练 BNN 不同 lookback：

```powershell
python -m src.experiments.train --config configs/models/bnn_1h.yaml
python -m src.experiments.train --config configs/models/bnn_4h.yaml
python -m src.experiments.train --config configs/models/bnn_8h.yaml
python -m src.experiments.train --config configs/models/bnn_24h.yaml
```

训练 baseline：

```powershell
python -m src.experiments.train --config configs/models/mlp_24h.yaml
python -m src.experiments.train --config configs/models/cnn_24h.yaml
python -m src.experiments.train --config configs/models/mc_dropout_24h.yaml
```

输出：

```text
outputs/runs/<model_name>/<timestamp>/
  config.yaml
  manifest.json
  metrics/train_history.csv
  logs/train.log
  models/best.pt         # PyTorch 后端
  models/best.pkl        # NumPy 后端
```

`train_history.csv` 会写出训练过程指标：

```text
train_rmse, train_nll
val_rmse, val_nll
epoch loss（PyTorch 后端）
```

训练时，控制台会按三段格式化打印：前面是 `Training Parameters`（模型、设备、lookback/horizon、窗口数、训练参数、输出目录），中间是 `Training Process`（开始/结束时间、用时、epoch loss），后面是 `Training Results`（总用时、验证集 RMSE/NLL、日志路径和模型文件路径）。训练入口不再负责最终 test 指标。

不同 lookback 必须分别训练，因为输入窗口长度不同，模型输入维度也不同。

## 统一评估与对比

对比命令不会训练模型。它会加载一个或多个已训练 run 的 best 模型，在统一 split（默认 `test`）上重新预测、计算指标、保存预测结果和图表。

先把 run 目录填入：

```text
configs/compare/main.yaml
configs/compare/lookback.yaml
```

然后运行：

```powershell
python -m src.experiments.compare --config configs/compare/main.yaml
python -m src.experiments.compare --config configs/compare/lookback.yaml
```

也可以不写 YAML，直接在命令行传入一个或多个 run；只传一个 run 也会生成汇总文件：

```powershell
python -m src.experiments.compare --run BNN-24h=outputs/runs/improved_bnn/你的run目录
python -m src.experiments.compare --run BNN-24h=outputs/runs/improved_bnn
python -m src.experiments.compare --name main --run BNN-24h=outputs/runs/improved_bnn --run CNN-24h=outputs/runs/cnn_baseline
```

如果 `--run` 的路径是模型根目录，例如 `outputs/runs/improved_bnn`，脚本会自动选择其中最新的时间戳 run 目录；YAML 中的 `runs[].path` 也支持同样写法。`compare_results.py` 仍保留为兼容入口。

输出：

```text
outputs/comparisons/<name>/<timestamp>/
  compare_config.yaml
  model_metrics.csv
  model_metrics.txt
  predictions/
  figures/
  report.md
```

## 模型说明

当前 PyTorch 模型在 `src/models/torch_models.py`。

- `improved_bnn`：三分支结构，分别处理 history、weather、direct，再融合预测；
- `mlp_baseline`：使用全部输入的 PyTorch 网络；
- `cnn_baseline`：只使用历史功率输入；
- `mc_dropout`：使用历史功率和未来天气输入，并启用 dropout。

`branch_dim` 只对分支结构有意义，当前会控制 history/weather/direct 分支输出维度：

```yaml
model:
  name: improved_bnn
  hidden_dim: 128
  branch_dim: 64
```

## 配置 include

模型配置可以引用数据配置：

```yaml
include:
  - ../data.yaml

data:
  lookback: 4

model:
  name: improved_bnn
```

`include` 中的配置先加载，当前文件中的字段会覆盖它。

## 测试

```powershell
pytest -q
```
