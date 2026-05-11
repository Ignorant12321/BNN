# 基于贝叶斯神经网络的分布式光伏出力概率预测

本项目用于研究分布式光伏电站未来 4 小时 `AC_POWER` 的概率预测。代码以 Plant 1 发电数据和气象传感器数据为实验对象，在 15 分钟时间粒度下构造监督学习样本，并用改进贝叶斯神经网络输出未来 16 步预测均值、方差和预测区间。

README 以当前代码实现为准，重点说明如何准备数据、调参、训练、评估和查看结果。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

准备数据并训练：

```bash
python -m src.prepare_data --config configs/default.yaml```
```

训练模型：

```bash
python -m src.train --config configs/default.yaml
```

自动调参：

```bash
python -m src.tune
```

把最新调参结果迁移到正式训练配置：

```bash
python -m src.apply_tuning
```

查看实验结果：

```bash
python visualizer/server.py
```

默认打开 `http://127.0.0.1:5177/`。如需换端口：

```bash
python visualizer/server.py --port 5178
```

运行测试：

```bash
pytest -q
```

## 推荐工作流

1. 用 `configs/default.yaml` 运行 `prepare_data`，固化清洗结果和时间切分。
2. 用 `configs/tuning.yaml` 运行 Optuna 调参。调参只看验证集，不用测试集选模型。
3. 如需换验证集指标，从已有 trial 中用 `src.select_tuning` 重新选择 best trial。
4. 用 `src.apply_tuning` 将 `best_params.json` 中的参数迁移到 `configs/default.yaml`。
5. 用 `configs/default.yaml` 正式训练，并评估验证集和测试集。
6. 写报告或论文时，最终性能优先引用正式训练输出中的测试集指标。

## 数据与任务

原始数据放在 `dataset/`：

```text
dataset/
├── Plant_1_Generation_Data.csv
└── Plant_1_Weather_Sensor_Data.csv
```

任务设置：

| 项目 | 设置 |
| --- | --- |
| 时间粒度 | 15 分钟 |
| 历史窗口 | 过去 16 步，即预测点前 4 小时 |
| 预测窗口 | 未来 16 步，即未来 4 小时 |
| 预测目标 | 电站级 `AC_POWER` |
| 主要模型 | `ImprovedBayesianPVNet` |
| 输出 | 未来 16 步预测均值、方差和预测区间 |

窗口定义：

```text
history: [t - 16, ..., t - 1]
direct : t - 1
weather: [t, ..., t + 15]
target : [t, ..., t + 15]
```

`direct` 必须是预测点前一刻的数据，不能使用预测点 `t` 或预测点之后的数据。

特征分组由 `src/features.py` 统一维护：

| 分组 | 特征 | 含义 |
| --- | --- | --- |
| `history` | `AC_POWER` | 预测点前 4 小时的光伏出力历史序列 |
| `weather` | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `hour` | 预测窗口天气序列和小时数 |
| `direct` | `last_ac_power` | 预测点前一刻 `AC_POWER` |
| `target` | `AC_POWER` | 未来 4 小时预测目标 |

论文语境下，`weather` 表示可由数值天气预报获得的未来天气特征。当前公开数据集没有真实 NWP 文件，因此实验中使用预测窗口内的真实气象观测值作为天气预报替代输入。论文写作时需要明确说明这是数据集限制下的模拟设定。

## 数据处理

主流程会完成：

1. 解析 `DATE_TIME`。
2. 将多个逆变器记录聚合为电站级时间序列。
3. 按时间合并气象数据。
4. 补齐 15 分钟规则时间轴并处理缺失值。
5. 添加 `hour`、周期时间特征、白天标记和 `last_ac_power`。
6. 按时间顺序切分 train / val / test。
7. 在每个子集内部构造滑动窗口，避免窗口跨越切分边界。
8. 只在训练集窗口上拟合 scaler，再用于验证集和测试集。

可先把清洗和时间切分单独固化为中间文件：

```bash
python -m src.prepare_data --config configs/default.yaml
```

输出：

```text
data/processed/
├── train.csv
├── val.csv
├── test.csv
└── split_info.json
```

之后训练会优先读取这些已处理文件。如果文件不存在，则自动回退到从原始 CSV 即席清洗和切分。滑动窗口仍在训练时根据 `lookback` 和 `horizon` 动态构造。

## 配置说明

主要配置文件：

| 文件 | 用途 |
| --- | --- |
| `configs/default.yaml` | 正式训练和测试集评估 |
| `configs/tuning.yaml` | Optuna 调参，不会自动继承 `default.yaml` |
| `configs/compare.yaml` | 模型对比实验 |

`default.yaml` 和 `tuning.yaml` 的大多数字段含义相同。区别是：`tuning.yaml` 每个 trial 会复制这份配置，然后用 `tuning.search_space` 中采样到的值覆盖部分模型和训练字段。

常用字段：

| 字段 | 含义 |
| --- | --- |
| `seed` | 随机种子 |
| `output_dir` | 输出根目录 |
| `data.processed_dir` | `prepare_data` 写出的 train/val/test 中间文件目录 |
| `data.train_ratio` / `data.val_ratio` | 按时间顺序切分训练集和验证集，剩余为测试集 |
| `data.lookback` | 历史输入窗口长度，当前 `16` 表示过去 4 小时 |
| `data.horizon` | 预测窗口长度，当前 `16` 表示未来 4 小时 |
| `model.hidden_dim` | 贝叶斯融合层隐藏维度 |
| `model.branch_dim` | history/weather 分支编码维度 |
| `model.prior_sigma` | 贝叶斯线性层权重先验标准差 |
| `training.device` | `auto` 会优先使用 CUDA，否则使用 CPU |
| `training.epochs` | 最多训练轮数，实际可能被 early stopping 提前停止 |
| `training.lr` | AdamW 学习率 |
| `training.kl_beta` | BNN KL 散度项权重 |
| `training.patience` | 验证损失连续多少轮未改善后 early stopping |
| `prediction.mc_samples` | 推理时 Monte Carlo 前向传播次数 |
| `evaluation.run_test` | 是否在训练结束后评估测试集，正式训练建议为 `true` |

`configs/tuning.yaml` 中额外包含：

| 字段 | 含义 |
| --- | --- |
| `tuning.study_name` | Optuna study 名称，同名 study 会复用历史 trial |
| `tuning.storage` | Optuna 持久化存储位置 |
| `tuning.load_if_exists` | storage 中已有同名 study 时是否继续加载 |
| `tuning.n_trials` | 目标总 trial 数，不是每次启动追加的数量 |
| `tuning.objective_metric` | 选择最佳 trial 的验证集指标，例如 `crps`、`rmse`、`nll`，数值越小越好 |
| `tuning.search_space.*` | `hidden_dim`、`branch_dim`、`lr`、`kl_beta` 的搜索范围 |

配置使用注意：

1. 改了 `default.yaml` 不会自动同步到 `tuning.yaml`，两份配置需要按实验目的分别维护。
2. 调参时 `model.hidden_dim`、`model.branch_dim`、`training.lr`、`training.kl_beta` 会被 `tuning.search_space` 覆盖。
3. 最终结果应来自测试集 `metrics/metrics.json`；调参和模型选择应看验证集 `metrics/validation_metrics.json`。

## 训练与评估

正式训练：

```bash
python -m src.train --config configs/default.yaml
```

为单次实验写备注，备注会保存到本次输出目录的 `note.txt`：

```bash
python -m src.train --config configs/default.yaml --note "best optuna config"
```

训练结束默认会自动评估验证集和测试集。若想对已有训练结果重新评估，例如调整 MC 采样次数或重新生成预测文件：

```bash
python -m src.evaluate_model --run-dir outputs/improved_bnn/YYYYMMDD-HHMMSS --split test
python -m src.evaluate_model --run-dir outputs/improved_bnn/YYYYMMDD-HHMMSS --split both --mc-samples 50
```

## 调参与参数迁移

自动调参：

```bash
python -m src.tune
```

一次调参会话会写入：

```text
outputs/tuning/YYYYMMDD-HHMMSS/
├── best_params.json
├── best_config.yaml
├── trials.csv
└── improved_bnn/<trial时间戳>/
```

`best_params.json`、`best_config.yaml` 和 `trials.csv` 位于调参会话根目录；每个 trial 的训练产物位于 `improved_bnn/<trial时间戳>/`。

`configs/tuning.yaml` 默认使用 SQLite 持久化 study：`outputs/tuning/optuna.db`。如果调参进程中断，再次运行 `python -m src.tune` 会通过同名 study 继续搜索已完成 trial 之后的配置。此时 `tuning.n_trials` 表示目标总 trial 数。

### 应用最新 best params

把最新调参会话的 `best_params.json` 迁移到正式训练配置：

```bash
python -m src.apply_tuning
```

该命令只迁移 Optuna 实际搜索的字段：

```text
model.hidden_dim
model.branch_dim
training.lr
training.kl_beta
```

常用命令：

```bash
python -m src.apply_tuning --objective crps
python -m src.apply_tuning --objective crps --yes
python -m src.apply_tuning --source outputs/tuning/YYYYMMDD-HHMMSS/best_params.json
python -m src.apply_tuning --no-color
```

`apply_tuning` 会先展示 source、target、objective 和参数变化，只有确认输入 `y` 或 `yes` 后才会写入 `configs/default.yaml`。如需在脚本中跳过确认，可加 `--yes`。

当 `--source latest` 时，`apply_tuning` 只检查最新的 tuning 会话。如果最新会话的 `objective_metric` 和 `--objective` 不一致，命令会拒绝执行，并提示先用 `src.select_tuning` 在最新会话里重新选择指标。它不会偷偷回退到旧的 tuning 会话。

### 从已有 trial 切换指标

如果已经完成一轮调参，但不想重新训练，又想改用另一个验证集指标选择 best trial：

```bash
python -m src.select_tuning --source latest --show
python -m src.select_tuning --source latest --metric rmse
python -m src.apply_tuning --objective rmse
```

职责边界：

| 命令 | 负责什么 | 会写什么 |
| --- | --- | --- |
| `src.select_tuning` | 查询已有 trial，并按验证集指标重选 best trial | 当前 tuning 会话的 `best_params.json` 和 `best_config.yaml` |
| `src.apply_tuning` | 把 `best_params.json` 中的参数迁移到正式训练配置 | `configs/default.yaml` |

`select_tuning` 和 `apply_tuning` 都会先展示将要发生的变化，只有确认输入 `y` 或 `yes` 后才会写文件；如需自动化执行，可加 `--yes`。两个命令默认使用克制的彩色输出；如果终端或日志不需要颜色，可加 `--no-color`。

## 可视化

启动本地服务：

```bash
python visualizer/server.py
```

浏览器打开终端输出的地址，默认是 `http://127.0.0.1:5177/`。

可视化页面支持：

| 功能 | 持久化位置 |
| --- | --- |
| 隐藏/显示 run | `visualizer/hidden-runs.json` |
| 编辑 run 备注 | 对应 run 目录下的 `note.txt` |

如果只临时查看静态页面，也可以用 Live Server 打开 `visualizer/index.html`；但需要保存隐藏状态和备注时，请使用 `python visualizer/server.py`。

## 输出与指标

正式训练结果写入：

```text
outputs/improved_bnn/YYYYMMDD-HHMMSS/
```

常用文件：

| 路径 | 内容 |
| --- | --- |
| `checkpoints/best_model.pt` | 验证集最优权重 |
| `checkpoints/last_model.pt` | 最后一个 epoch 的权重 |
| `logs/train.log` | 训练日志、best epoch、early stopping 和指标摘要 |
| `metrics/validation_metrics.json` | 验证集整体指标 |
| `metrics/metrics.json` | 测试集整体指标 |
| `metrics/point_metrics.csv` | 测试集按预测步长统计的点预测指标 |
| `metrics/probabilistic_metrics.csv` | 测试集概率预测指标表格 |
| `predictions/test_predictions.csv` | 测试集逐步预测 |
| `predictions/uncertainty_samples.npy` | 测试集 MC 不确定性采样结果 |
| `figures/loss_curve.png` | 训练和验证损失曲线 |
| `figures/prediction_interval_90.png` | 90% 预测区间 |
| `figures/prediction_interval_95.png` | 95% 预测区间 |

看结果时建议：

1. 训练过程中看 `logs/train.log` 和 `figures/loss_curve.png`，确认训练损失和验证损失是否正常下降，是否触发 early stopping。
2. 模型选择和调参看 `metrics/validation_metrics.json`。
3. 最终实验结果看 `metrics/metrics.json`。
4. 若 `evaluation.run_test: false`，程序只会跑验证集，不会生成测试集指标文件。

指标含义：

| 指标 | 含义 | 越大越好？ |
| --- | --- | --- |
| `mae` | 平均绝对误差，单位与 `AC_POWER` 相同 | 否 |
| `rmse` | 均方根误差，对大误差更敏感 | 否 |
| `nrmse` | 用真实值范围归一化后的 RMSE | 否 |
| `smape` | 对称平均绝对百分比误差，夜间接近 0 时较敏感 | 否 |
| `crps` | 连续排名概率分数，综合评价概率预测准确性、校准性和锐度 | 否 |
| `nll` | 高斯负对数似然，综合评价均值和方差是否合理 | 否 |
| `picp_90` / `picp_95` | 90% / 95% 预测区间覆盖率 | 越接近目标覆盖率越好 |
| `pinaw_90` / `pinaw_95` | 90% / 95% 预测区间归一化平均宽度 | 在 PICP 达标前提下越小越好 |

点预测主要看 `mae`、`rmse`、`nrmse`，数值越低越好。概率预测优先看 `crps`，并同时检查 `picp` 和 `pinaw`。理想情况下，`picp_90` 接近 0.90、`picp_95` 接近 0.95，并且 `pinaw` 不要过大。

## 模型结构

主模型位于 `src/models/improved_bnn.py`。模型按数据来源分三路处理，再在融合层中统一建模。

```text
weather [batch, 16, 4]
  -> ForecastWeatherMLPBranch
  -> Linear(4 * 16 -> 32)
  -> Linear(32 -> 64)
  -> Linear(64 -> branch_dim)

history [batch, 16, 1]
  -> HistoryCNNBranch
  -> Conv1d(kernel_size=5)
  -> AvgPool1d(kernel_size=5)
  -> Conv1d(kernel_size=5)
  -> AdaptiveAvgPool1d(1)
  -> Linear(branch_dim -> branch_dim)

direct [batch, 1]
  -> 直接拼接进融合层

fusion
  -> BayesianLinear
  -> BayesianLinear
  -> mean_head: BayesianLinear(hidden_dim -> 16)
  -> log_var_head: BayesianLinear(hidden_dim -> 16)
```

输入和输出：

| 名称 | Shape | 作用 |
| --- | --- | --- |
| `weather` | `[batch, 16, 4]` | 未来 16 个步长的天气和小时特征 |
| `history` | `[batch, 16, 1]` | 预测点前 4 小时历史 `AC_POWER` |
| `direct` | `[batch, 1]` | 预测点前一刻 `last_ac_power` |
| `mean_head` | `[batch, 16]` | 未来 16 步预测均值 |
| `log_var_head` | `[batch, 16]` | 未来 16 步对数方差 |

`BayesianLinear` 将每个权重和偏置建模为高斯分布。模型学习后验分布参数 `mu` 和 `rho`，其中 `sigma = softplus(rho)`。训练时通过重参数化采样权重：

```text
weight = mu + sigma * eps
```

损失函数采用近似 ELBO：

```text
loss = Gaussian NLL(mean, log_var, target) + beta * KL / num_batches
```

推理时执行多次 Monte Carlo 前向传播。不同权重采样得到的预测差异表示模型不确定性，`log_var_head` 输出的方差表示数据噪声不确定性，二者合并后得到最终 `y_std` 和预测区间。

## 代码结构

```text
src/
├── data.py                # CSV 读取、聚合与合并
├── prepare_data.py        # 清洗原始数据并写出 train/val/test 切分
├── features.py            # 特征工程与特征分组
├── dataset.py             # 时间切分、窗口构造和 Dataset
├── losses.py              # Gaussian NLL 与 ELBO 风格损失
├── metrics.py             # 点预测与概率预测指标
├── evaluate.py            # 指标汇总
├── evaluation_pipeline.py # 训练后评估、预测导出和图像生成
├── predict.py             # MC 推理与区间计算
├── visualization.py       # 实验图像绘制
├── train.py               # 主训练入口
├── evaluate_model.py      # 独立评估已有训练结果
├── tune.py                # Optuna 调参入口
├── select_tuning.py       # 从已有 trial 中按指标重选 best trial
├── apply_tuning.py        # 将 best_params.json 迁移到 default.yaml
└── models/
    ├── bayesian_layers.py
    ├── branches.py
    ├── improved_bnn.py
    └── baselines.py
```

## 论文写作说明

论文中建议按以下方式描述本实现：

1. 使用过去 4 小时 `AC_POWER` 作为历史出力序列。
2. 使用未来 4 小时天气序列作为条件输入，字段包括辐照度、环境温度、组件温度和小时。
3. 将预测点前一刻 `AC_POWER` 作为直接输入，强调其不是预测点之后的数据。
4. 三路输入融合后进入贝叶斯全连接层，同时输出预测均值和方差。
5. 通过 Monte Carlo 前向传播估计模型不确定性，并结合输出方差形成预测区间。
6. 当前天气输入由真实观测模拟 NWP，后续接入真实数值天气预报后可进一步验证部署性能。
