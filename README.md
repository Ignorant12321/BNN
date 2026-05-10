# 基于贝叶斯神经网络的分布式光伏出力概率预测

本项目用于研究分布式光伏电站未来 4 小时 `AC_POWER` 的概率预测。代码以 Plant 1 发电数据和气象传感器数据为实验对象，在 15 分钟时间粒度下构造监督学习样本，并用改进贝叶斯神经网络输出预测均值和不确定性区间。

当前实现以用户给定论文结构为准，README 只作为代码说明，不再沿用旧版本中错误的四分支解释。

## 任务设置

| 项目 | 设置 |
| --- | --- |
| 时间粒度 | 15 分钟 |
| 历史窗口 | 过去 16 步，即预测点前 4 小时 |
| 预测窗口 | 未来 16 步，即未来 4 小时 |
| 预测目标 | 电站级 `AC_POWER` |
| 主要模型 | `ImprovedBayesianPVNet` |
| 输出 | 未来 16 步预测均值、方差和预测区间 |

窗口定义如下：

```text
history: [t - 16, ..., t - 1]
direct : t - 1
weather: [t, ..., t + 15]
target : [t, ..., t + 15]
```

其中 `direct` 必须是预测点前一刻的数据，不能使用预测点 `t` 或预测点之后的数据。

## 输入特征

特征分组由 `src/features.py` 统一维护。

| 分组 | 特征 | 含义 |
| --- | --- | --- |
| `history` | `AC_POWER` | 预测点前 4 小时的光伏出力历史序列 |
| `weather` | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `hour` | 预测窗口天气序列和小时数 |
| `direct` | `last_ac_power` | 预测点前一刻 `AC_POWER` |
| `target` | `AC_POWER` | 未来 4 小时预测目标 |

论文语境下，`weather` 表示可由数值天气预报获得的未来天气特征。当前公开数据集没有真实 NWP 文件，因此实验中使用预测窗口内的真实气象观测值作为天气预报替代输入。论文写作时需要明确说明这是数据集限制下的模拟设定。

## 数据处理

原始数据放在 `dataset/` 目录：

```text
dataset/
├── Plant_1_Generation_Data.csv
└── Plant_1_Weather_Sensor_Data.csv
```

主流程会完成：

1. 解析 `DATE_TIME`。
2. 将多个逆变器记录聚合为电站级时间序列。
3. 按时间合并气象数据。
4. 补齐 15 分钟规则时间轴并处理缺失值。
5. 添加 `hour`、周期时间分析特征、白天标记和 `last_ac_power`。
6. 按时间顺序切分 train / val / test。
7. 在每个子集内部构造滑动窗口，避免窗口跨越切分边界。
8. 只在训练集窗口上拟合 scaler，再用于验证集和测试集。

也可以先把清洗和时间切分单独固化为中间文件：

```bash
python -m src.prepare_data --config configs/default.yaml
```

该命令会写出：

```text
data/processed/
├── train.csv
├── val.csv
├── test.csv
└── split_info.json
```

之后训练会优先读取这些已处理文件；如果文件不存在，则自动回退到从原始 CSV 即席清洗和切分。滑动窗口仍在训练时根据 `lookback` 和 `horizon` 动态构造，方便比较不同历史窗口和预测窗口设置。

## 模型结构

主模型位于 `src/models/improved_bnn.py`。它不是把所有特征简单拼成一个长向量，而是先按数据来源分成三路处理，再在融合层中统一建模。这样做的原因是：历史功率是时间序列，未来天气是预测窗口内的条件序列，而预测点前一刻功率是一个很强的短期状态量，三类输入的数据形态不同。

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

三路输入的含义如下：

| 输入 | Shape | 作用 |
| --- | --- | --- |
| `weather` | `[batch, 16, 4]` | 未来 16 个 15 分钟步长的天气和小时特征，用 MLP 提取预测窗口内的条件信息 |
| `history` | `[batch, 16, 1]` | 预测点前 4 小时的历史 `AC_POWER`，用 1D-CNN 提取局部波动、爬坡和下降趋势 |
| `direct` | `[batch, 1]` | 预测点前一刻 `last_ac_power`，直接拼接进融合层，提供最近状态 |

融合后的向量进入两层 `BayesianLinear`，最后分成两个输出头：

| 输出头 | Shape | 含义 |
| --- | --- | --- |
| `mean_head` | `[batch, 16]` | 未来 16 步的预测均值，即点预测结果 |
| `log_var_head` | `[batch, 16]` | 未来 16 步的对数方差，用于描述每个预测点自身的噪声不确定性 |

`BayesianLinear` 与普通 `Linear` 的区别是：普通层的权重是一个确定值，而这里每个权重和偏置都被建模为高斯分布。模型实际学习的是后验分布参数 `mu` 和 `rho`，其中 `sigma = softplus(rho)`。训练时每次前向传播都会用重参数化方式采样一组权重：

```text
weight = mu + sigma * eps
```

损失函数采用近似 ELBO 形式：

```text
loss = Gaussian NLL(mean, log_var, target) + beta * KL / num_batches
```

其中 Gaussian NLL 约束预测均值和方差贴近真实功率，KL 项约束贝叶斯层的后验不要偏离先验太远。推理时模型会执行多次 Monte Carlo 前向传播；不同权重采样得到的预测差异表示模型不确定性，`log_var_head` 输出的方差表示数据噪声不确定性，二者合并后得到最终的 `y_std` 和预测区间。

## 配置

主要配置文件：

| 文件 | 用途 |
| --- | --- |
| `configs/default.yaml` | 正式训练 |
| `configs/tuning.yaml` | Optuna 调参；它是一份独立调参配置，不会自动继承 `default.yaml` |
| `configs/compare.yaml` | 模型对比实验 |

`default.yaml` 和 `tuning.yaml` 的大多数字段含义相同。区别是：`default.yaml` 面向最终训练和测试集评估；`tuning.yaml` 面向 Optuna 搜索，每个 trial 会复制这份配置，然后用 `tuning.search_space` 中采样到的值覆盖部分模型和训练字段。

通用字段：

| 字段 | 含义 |
| --- | --- |
| `seed` | 随机种子，用于提高实验可复现性 |
| `output_dir` | 输出根目录；正式训练默认写入 `outputs/improved_bnn/时间戳/` |

`data` 字段：

| 字段 | 含义 |
| --- | --- |
| `generation_path` | 发电数据 CSV 路径 |
| `weather_path` | 气象数据 CSV 路径 |
| `processed_dir` | `prepare_data` 写出的 train/val/test 中间文件目录；训练时若存在这些文件会优先读取 |
| `fill_missing` | 是否补齐 15 分钟规则时间轴并插值缺失值 |
| `train_ratio` | 按时间顺序切出的训练集比例 |
| `val_ratio` | 按时间顺序切出的验证集比例；剩余部分为测试集 |
| `lookback` | 历史输入窗口长度。当前 `16` 表示过去 4 小时 |
| `horizon` | 预测窗口长度。当前 `16` 表示未来 4 小时 |

`model` 字段：

| 字段 | 含义 |
| --- | --- |
| `name` | 模型名称；当前主流程使用 `improved_bnn` |
| `hidden_dim` | 贝叶斯融合层的隐藏维度 |
| `branch_dim` | history/weather 分支编码后的维度 |
| `prior_sigma` | 贝叶斯线性层权重先验分布的标准差 |

`training` 字段：

| 字段 | 含义 |
| --- | --- |
| `device` | 训练设备，`auto` 会优先使用 CUDA，否则使用 CPU |
| `epochs` | 最多训练轮数；实际可能被 early stopping 提前停止 |
| `batch_size` | DataLoader batch 大小 |
| `num_workers` | DataLoader 子进程数；Windows + CUDA 下调参会强制改为 `0` |
| `pin_memory` | CUDA 训练时是否启用 pinned memory 加速数据搬运 |
| `persistent_workers` | `num_workers > 0` 时是否保持 DataLoader worker 常驻 |
| `amp` | CUDA 上是否启用自动混合精度 |
| `cudnn_benchmark` | CUDA 上是否允许 cuDNN 为固定输入形状选择更快算法 |
| `lr` | AdamW 学习率；在调参 trial 中会被 `tuning.search_space.lr` 的采样值覆盖 |
| `weight_decay` | AdamW 权重衰减 |
| `kl_beta` | BNN KL 散度项权重；在调参 trial 中会被 `tuning.search_space.kl_beta` 的采样值覆盖 |
| `patience` | 验证损失连续多少轮未改善后 early stopping |

`prediction` 字段：

| 字段 | 含义 |
| --- | --- |
| `mc_samples` | 推理时 Monte Carlo 前向传播次数；越大不确定性估计越稳定，但评估越慢 |
| `plot.prefer_daylight` | 未指定固定时间段时，是否优先选择白天出力片段画预测区间 |
| `plot.daylight_threshold` | 判断白天片段的真实功率阈值 |
| `plot.max_points` | 预测区间图最多绘制多少个点 |
| `plot.start_time` / `plot.end_time` | 指定预测区间图展示的时间段 |

`evaluation` 字段：

| 字段 | 含义 |
| --- | --- |
| `run_test` | 是否在训练结束后评估测试集。正式训练建议为 `true`；调参时由代码强制设为 `false`，避免用测试集选模型 |

`tuning` 字段仅用于 `configs/tuning.yaml`：

| 字段 | 含义 |
| --- | --- |
| `study_name` | Optuna study 名称；同名 study 会复用历史 trial |
| `storage` | Optuna 持久化存储位置，例如 `sqlite:///outputs/tuning/optuna.db` |
| `load_if_exists` | storage 中已有同名 study 时是否继续加载 |
| `n_trials` | 目标总 trial 数，不是每次启动追加的数量 |
| `search_space.hidden_dim` | `model.hidden_dim` 的候选列表 |
| `search_space.branch_dim` | `model.branch_dim` 的候选列表 |
| `search_space.lr.low/high/log` | 学习率搜索下界、上界以及是否按对数尺度采样 |
| `search_space.kl_beta.low/high/log` | KL 权重搜索下界、上界以及是否按对数尺度采样 |

使用配置时建议注意三点：

1. 改了 `default.yaml` 不会自动同步到 `tuning.yaml`，两份配置需要按实验目的分别维护。
2. 调参时 `model.hidden_dim`、`model.branch_dim`、`training.lr`、`training.kl_beta` 会被 `tuning.search_space` 覆盖；它们保留在 `tuning.yaml` 里主要是为了让这份文件仍可作为普通训练配置使用。
3. 最终论文或报告结果应来自测试集 `metrics/metrics.json`；调参和模型选择应看验证集 `metrics/validation_metrics.json`。

## 运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行测试：

```bash
pytest -q
```

推荐工作流：

1. 先运行 `prepare_data` 固化清洗结果和时间切分。
2. 用 `configs/tuning.yaml` 运行 Optuna 调参；调参只看验证集，不使用测试集选模型。
3. 用 `src.apply_tuning` 将最佳 `hidden_dim`、`branch_dim`、`lr`、`kl_beta` 迁移到 `configs/default.yaml`。
4. 用 `configs/default.yaml` 做正式训练，并评估测试集。
5. 写报告或论文时引用正式训练输出中的测试集指标。

第一次运行建议先准备数据，再训练：

```bash
python -m src.prepare_data --config configs/default.yaml
python -m src.train --config configs/default.yaml
```

`prepare_data` 会完成清洗和 train/val/test 时间切分。后续如果原始数据、切分比例和清洗逻辑不变，可以直接训练：

```bash
python -m src.train --config configs/default.yaml
```

也可以为单次实验写备注，备注会保存到本次输出目录的 `note.txt`：

```bash
python -m src.train --config configs/default.yaml --note "best optuna config"
```

训练结束默认会自动评估验证集和测试集。若想对某次训练结果重新评估，例如调整 MC 采样次数或重新生成预测文件，可运行：

```bash
python -m src.evaluate_model --run-dir outputs/improved_bnn/YYYYMMDD-HHMMSS --split test
python -m src.evaluate_model --run-dir outputs/improved_bnn/YYYYMMDD-HHMMSS --split both --mc-samples 50
```

自动调参：

```bash
python -m src.tune
```

一次调参会话会写入 `outputs/tuning/YYYYMMDD-HHMMSS/`。其中每个 trial 的训练产物位于
`outputs/tuning/YYYYMMDD-HHMMSS/improved_bnn/<trial时间戳>/`，调参汇总文件
`best_params.json`、`trials.csv`、`best_config.yaml` 位于同一个调参会话目录根部。
`configs/tuning.yaml` 默认启用 SQLite 持久化 study：`outputs/tuning/optuna.db`。
如果调参进程中断，再次运行 `python -m src.tune` 会通过同名 study 继续搜索已完成 trial 之后的配置。
此时 `tuning.n_trials` 表示目标总 trial 数，而不是每次重启追加的 trial 数。
Optuna 的搜索范围配置在 `configs/tuning.yaml` 的 `tuning.search_space` 中；例如
`lr.low`/`lr.high` 控制学习率范围，`kl_beta.low`/`kl_beta.high` 控制 BNN 的 KL 权重范围。
`training.epochs` 和 `training.patience` 则控制每个 trial 最多训练多少轮以及 early stopping 的耐心值。

把最新调参结果迁移到正式训练配置：

```bash
python -m src.apply_tuning
```

该命令默认读取最新 `outputs/tuning/*/best_params.json`，只迁移 Optuna 实际搜索的
`model.hidden_dim`、`model.branch_dim`、`training.lr` 和 `training.kl_beta`，并打印每个字段的
旧值、新值和是否发生变化。也可以显式指定来源或先预览：

```bash
python -m src.apply_tuning --source outputs/tuning/YYYYMMDD-HHMMSS/best_params.json
python -m src.apply_tuning --source outputs/tuning/YYYYMMDD-HHMMSS/best_params.json --dry-run
```

## 输出

训练结果会写入：

```text
outputs/improved_bnn/YYYYMMDD-HHMMSS/
```

常用文件包括：

| 路径 | 内容 |
| --- | --- |
| `checkpoints/best_model.pt` | 验证集最优权重 |
| `metrics/metrics.json` | 测试集指标 |
| `metrics/validation_metrics.json` | 验证集指标 |
| `metrics/point_metrics.csv` | 测试集按预测步长统计的点预测指标 |
| `metrics/probabilistic_metrics.csv` | 测试集概率预测指标表格 |
| `predictions/test_predictions.csv` | 测试集逐步预测 |
| `predictions/uncertainty_samples.npy` | 测试集 MC 不确定性采样结果 |
| `figures/loss_curve.png` | 训练和验证损失曲线 |
| `figures/prediction_interval_90.png` | 90% 预测区间 |
| `figures/prediction_interval_95.png` | 95% 预测区间 |

看结果时建议按下面顺序：

1. 训练过程中看 `logs/train.log` 和 `figures/loss_curve.png`，确认训练损失和验证损失是否正常下降，是否触发 early stopping。
2. 选模型或调参时看 `metrics/validation_metrics.json`。这个文件来自验证集，用于判断当前训练过程中的模型泛化表现，`best_model.pt` 也是按验证损失保存的。
3. 写最终实验结果时优先看 `metrics/metrics.json`。这个文件来自测试集，更适合作为论文或报告中的最终性能结果。
4. 如果配置里 `evaluation.run_test: false`，程序只会跑验证集，此时只生成 `validation_metrics.json`，不会生成测试集的 `metrics.json`、`point_metrics.csv` 和 `probabilistic_metrics.csv`。

截图中的四个指标文件可以这样理解：

| 文件 | 数据集 | 怎么看 |
| --- | --- | --- |
| `validation_metrics.json` | 验证集 | 用于模型选择、调参和判断训练是否过拟合；不是最终测试结果 |
| `metrics.json` | 测试集 | 最终整体指标，论文表格一般优先引用这个文件 |
| `point_metrics.csv` | 测试集 | 按 horizon 分开的点预测指标，包含未来第 1 到第 16 步的 MAE/RMSE，可用于画“预测步长越远误差如何变化” |
| `probabilistic_metrics.csv` | 测试集 | 测试集概率预测指标的 CSV 版本，内容与 `metrics.json` 中的概率指标对应，方便导入表格软件 |

`metrics.json` 和 `validation_metrics.json` 中字段含义：

| 指标 | 含义 | 越大越好？ |
| --- | --- | --- |
| `mae` | 平均绝对误差，单位与 `AC_POWER` 相同 | 否 |
| `rmse` | 均方根误差，对大误差更敏感 | 否 |
| `nrmse` | 用真实值范围归一化后的 RMSE，便于不同实验对比 | 否 |
| `smape` | 对称平均绝对百分比误差，夜间接近 0 时会比较敏感 | 否 |
| `nll` | 高斯负对数似然，综合评价均值和方差是否合理 | 否 |
| `picp_90` / `picp_95` | 90% / 95% 预测区间覆盖率，表示真实值落入区间的比例 | 越接近目标覆盖率越好 |
| `pinaw_90` / `pinaw_95` | 90% / 95% 预测区间归一化平均宽度 | 在 PICP 达标前提下越小越好 |

简单判断规则：点预测主要看 `mae`、`rmse`、`nrmse`，数值越低越好；概率预测不能只看区间越窄，还要同时看 `picp`。理想情况下，`picp_90` 接近 0.90、`picp_95` 接近 0.95，并且 `pinaw` 不要过大。如果 `picp` 很低，说明区间太窄、真实值经常落在区间外；如果 `picp` 很高但 `pinaw` 也很大，说明区间过宽，预测虽然保守但信息量不足。

## 代码结构

```text
src/
├── data.py              # CSV 读取、聚合与合并
├── prepare_data.py      # 清洗原始数据并写出 train/val/test 切分
├── features.py          # 特征工程与特征分组
├── dataset.py           # 时间切分、窗口构造和 Dataset
├── losses.py            # Gaussian NLL 与 ELBO 风格损失
├── metrics.py           # 点预测与概率预测指标
├── evaluate.py          # 指标汇总
├── evaluation_pipeline.py # 训练后评估、预测导出和图像生成
├── predict.py           # MC 推理与区间计算
├── visualization.py     # 实验图像绘制
├── train.py             # 主训练入口
├── evaluate_model.py    # 独立评估已有训练结果
├── tune.py              # Optuna 调参入口
└── models/
    ├── bayesian_layers.py
    ├── branches.py
    ├── improved_bnn.py
    └── baselines.py
```

## 写作说明

论文中建议按以下方式描述本实现：

1. 使用过去 4 小时 `AC_POWER` 作为历史出力序列。
2. 使用未来 4 小时天气序列作为条件输入，字段包括辐照度、环境温度、组件温度和小时。
3. 将预测点前一刻 `AC_POWER` 作为直接输入，强调其不是预测点之后的数据。
4. 三路输入融合后进入贝叶斯全连接层，同时输出预测均值和方差。
5. 通过 Monte Carlo 前向传播估计模型不确定性，并结合输出方差形成预测区间。
6. 当前天气输入由真实观测模拟 NWP，后续接入真实数值天气预报后可进一步验证部署性能。
