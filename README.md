# 基于贝叶斯神经网络的分布式光伏出力概率预测

本项目研究分布式光伏电站未来 4 小时 `AC_POWER` 的概率预测。代码以 Plant 1 发电数据和气象传感器数据为实验对象，在 15 分钟时间粒度下构造监督学习样本，并使用改进贝叶斯神经网络输出未来 16 步的预测均值、方差和预测区间。

当前实现覆盖数据准备、模型训练、Optuna 调参、验证/测试集评估、结果可视化和自动化测试。README 以当前代码为准，说明如何复现实验、查看输出和理解工程目录。

## 项目概览

| 项目 | 设置 |
| --- | --- |
| 时间粒度 | 15 分钟 |
| 历史窗口 | 过去 16 步，即预测点前 4 小时 |
| 预测窗口 | 未来 16 步，即未来 4 小时 |
| 预测目标 | 电站级 `AC_POWER` |
| 主模型 | `ImprovedBayesianPVNet` |
| 输出 | 未来 16 步预测均值、方差、90%/95% 预测区间 |
| 调参工具 | Optuna |
| 可视化入口 | `visualizer/server.py` |

窗口定义：

```text
history: [t - 16, ..., t - 1]
direct : t - 1
weather: [t, ..., t + 15]
target : [t, ..., t + 15]
```

`direct` 只使用预测点前一刻的数据，不能使用预测点 `t` 或之后的数据。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

准备数据：

```bash
python -m src.prepare_data --config configs/default.yaml
```

训练并评估模型：

```bash
python -m src.train --config configs/default.yaml
```

查看实验结果：

```bash
python visualizer/server.py
```

默认访问地址为 `http://127.0.0.1:5177/`。如需换端口：

```bash
python visualizer/server.py --port 5178
```

运行测试：

```bash
pytest -q
```

## 推荐工作流

1. 用 `configs/default.yaml` 运行 `prepare_data`，固化清洗结果和时间切分。
2. 用 `configs/tuning.yaml` 运行 Optuna 调参。调参只使用验证集，不用测试集选模型。
3. 如需更换验证集选择指标，用 `src.select_tuning` 从已有 trial 中重新选择 best trial。
4. 用 `src.apply_tuning` 将 `best_params.json` 中的参数迁移到 `configs/default.yaml`。
5. 用 `configs/default.yaml` 正式训练，并评估验证集和测试集。
6. 写报告或论文时，最终性能优先引用正式训练输出中的测试集指标。

常用命令：

```bash
python -m src.prepare_data --config configs/default.yaml
python -m src.tune
python -m src.select_tuning --source latest --show
python -m src.select_tuning --source latest --metric crps
python -m src.apply_tuning --objective crps
python -m src.train --config configs/default.yaml --note "final config"
python -m src.compare_models --config configs/compare.yaml
```

## 工程目录

```text
BNN/
├── configs/                 # 实验配置
│   ├── default.yaml          # 正式训练和测试集评估
│   ├── tuning.yaml           # Optuna 调参配置
│   └── compare.yaml          # 模型对比实验配置
├── dataset/                 # 原始 Plant 1 数据
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
├── data/
│   └── processed/            # prepare_data 输出的 train/val/test 切分
├── src/                      # 核心 Python 代码
│   ├── data.py               # CSV 读取、逆变器聚合、气象合并
│   ├── prepare_data.py       # 数据清洗和时间切分入口
│   ├── features.py           # 特征工程与特征分组
│   ├── dataset.py            # 时间切分、滑动窗口和 Dataset
│   ├── train.py              # 主训练入口
│   ├── compare_models.py     # 模型对比实验入口
│   ├── evaluate_model.py     # 重新评估已有 run
│   ├── evaluation_pipeline.py # 评估、预测导出和图像生成
│   ├── predict.py            # MC 推理与预测区间计算
│   ├── losses.py             # Gaussian NLL 与 ELBO 风格损失
│   ├── metrics.py            # 点预测与概率预测指标
│   ├── tune.py               # Optuna 调参入口
│   ├── select_tuning.py      # 从已有 trial 中重选 best trial
│   ├── apply_tuning.py       # 将调参结果迁移到 default.yaml
│   └── models/
│       ├── bayesian_layers.py
│       ├── branches.py
│       ├── improved_bnn.py
│       └── baselines.py
├── visualizer/               # 本地实验结果查看页面
│   ├── server.py
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/                    # Pytest 测试
├── notebooks/                # 数据检查、特征分析和预测可视化 notebook
├── outputs/                  # 训练、调参和评估产物
├── requirements.txt
└── README.md
```

## 数据与特征

原始数据放在 `dataset/`：

```text
dataset/
├── Plant_1_Generation_Data.csv
└── Plant_1_Weather_Sensor_Data.csv
```

数据处理流程：

1. 解析 `DATE_TIME`。
2. 将多个逆变器记录按时间聚合为电站级时间序列。
3. 按时间戳合并气象传感器数据。
4. 补齐 15 分钟规则时间轴并处理缺失值。
5. 添加 `hour`、周期时间特征、白天标记和 `last_ac_power`。
6. 按时间顺序切分 train / val / test。
7. 在每个子集内部构造滑动窗口，避免窗口跨越切分边界。
8. 只在训练集窗口上拟合 scaler，再用于验证集和测试集。

运行数据准备后会写出：

```text
data/processed/
├── train.csv
├── val.csv
├── test.csv
└── split_info.json
```

训练时会优先读取 `data/processed/` 中的固定切分。如果文件不存在，则自动从原始 CSV 即席清洗和切分。滑动窗口仍在训练时根据 `lookback` 和 `horizon` 动态构造。

特征分组由 `src/features.py` 统一维护：

| 分组 | 特征 | 含义 |
| --- | --- | --- |
| `history` | `AC_POWER` | 预测点前 4 小时的光伏出力历史序列 |
| `weather` | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `hour` | 预测窗口天气序列和小时数 |
| `direct` | `last_ac_power` | 预测点前一刻 `AC_POWER` |
| `target` | `AC_POWER` | 未来 4 小时预测目标 |

论文语境下，`weather` 表示可由数值天气预报获得的未来天气特征。当前公开数据集没有真实 NWP 文件，因此实验中使用预测窗口内的真实气象观测值作为天气预报替代输入。论文写作时需要明确说明这是数据集限制下的模拟设定。

## 配置说明

主要配置文件：

| 文件 | 用途 |
| --- | --- |
| `configs/default.yaml` | 正式训练和测试集评估 |
| `configs/tuning.yaml` | Optuna 调参，不会自动继承 `default.yaml` |
| `configs/compare.yaml` | 模型对比实验配置 |

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

模型对比实验：

```bash
python -m src.compare_models --config configs/compare.yaml
```

当前 `configs/compare.yaml` 默认对比以下模型：

| 模型名 | 简要说明 |
| --- | --- |
| `improved_bnn` | 主模型。历史功率 CNN 分支、未来天气 MLP 分支和上一时刻功率直接输入融合后，使用 BayesianLinear 输出均值和方差。 |
| `mlp_baseline` | 简单 MLP 基线。将历史功率、未来天气和直接输入展平后输入全连接网络，用于检验复杂分支结构是否有收益。 |
| `cnn_baseline` | 历史序列 CNN 基线。只使用过去功率序列预测未来出力，用于衡量未来天气输入带来的增益。 |
| `mc_dropout` | MC Dropout 概率基线。使用 CNN + MLP 分支结构，并在推理时通过 dropout 多次前向传播估计模型不确定性。 |

对比实验会按 `compare.models` 依次训练模型。每次运行会创建一个独立时间戳目录，汇总表和各模型完整训练产物都会放在其中：

```text
outputs/compare/YYYYMMDD-HHMMSS/
├── model_metrics.csv
├── improved_bnn/YYYYMMDD-HHMMSS/
├── mlp_baseline/YYYYMMDD-HHMMSS/
├── cnn_baseline/YYYYMMDD-HHMMSS/
└── mc_dropout/YYYYMMDD-HHMMSS/
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

`configs/tuning.yaml` 默认使用 SQLite 持久化 study：`outputs/tuning/optuna.db`。如果调参进程中断，再次运行 `python -m src.tune` 会通过同名 study 继续搜索已完成 trial 之后的配置。此时 `tuning.n_trials` 表示目标总 trial 数，不是每次启动追加的数量。

应用最新调参结果：

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

如果已经完成一轮调参，但想改用另一个验证集指标选择 best trial：

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

`select_tuning` 和 `apply_tuning` 都会先展示将要发生的变化，只有确认输入 `y` 或 `yes` 后才会写文件；如需自动化执行，可加 `--yes`。如果终端或日志不需要颜色，可加 `--no-color`。

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

如果只临时查看静态页面，也可以用 Live Server 打开 `visualizer/index.html`。需要保存隐藏状态和备注时，请使用 `python visualizer/server.py`。

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

主模型位于 `src/models/improved_bnn.py`，类名为 `ImprovedBayesianPVNet`。

代码对应的分层结构如下：

<table>
  <thead>
    <tr>
      <th>层数</th>
      <th>第一部分：未来天气 MLP</th>
      <th>第二部分：历史功率 CNN</th>
      <th>第三部分：直接输入</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>输入层<br><code>weather [B, 16, 4]</code></td>
      <td>输入层<br><code>history [B, 16, 1]</code></td>
      <td>-</td>
    </tr>
    <tr>
      <td>2</td>
      <td>全连接层<br><code>Linear(4 * 16, 32)</code></td>
      <td>1D 卷积层<br><code>Conv1d(1, branch_dim, kernel_size=5)</code></td>
      <td>-</td>
    </tr>
    <tr>
      <td>3</td>
      <td>全连接层<br><code>Linear(32, 64)</code></td>
      <td>平均池化层<br><code>AvgPool1d(kernel_size=5)</code></td>
      <td>-</td>
    </tr>
    <tr>
      <td>4</td>
      <td>全连接层<br><code>Linear(64, branch_dim)</code></td>
      <td>1D 卷积层<br><code>Conv1d(branch_dim, branch_dim, kernel_size=5)</code></td>
      <td>-</td>
    </tr>
    <tr>
      <td>5</td>
      <td>-</td>
      <td>全局平均池化层<br><code>AdaptiveAvgPool1d(1)</code></td>
      <td>输入层<br><code>direct [B, 1]</code></td>
    </tr>
    <tr>
      <td>6</td>
      <td>-</td>
      <td>投影层<br><code>Linear(branch_dim, branch_dim)</code></td>
      <td>-</td>
    </tr>
    <tr>
      <td>7</td>
      <td colspan="3">合并层：<code>concat(weather_code, history_code, direct)</code></td>
    </tr>
    <tr>
      <td>8</td>
      <td colspan="3">概率全连接层：<code>BayesianLinear(branch_dim * 2 + 1, hidden_dim)</code></td>
    </tr>
    <tr>
      <td>9</td>
      <td colspan="3">概率全连接层：<code>BayesianLinear(hidden_dim, hidden_dim)</code></td>
    </tr>
    <tr>
      <td>10</td>
      <td colspan="3">输出层：<code>mean_head</code> 和 <code>log_var_head</code>，均为 <code>BayesianLinear(hidden_dim, 16)</code></td>
    </tr>
  </tbody>
</table>

```text
history: 过去功率序列 [B, 16, 1]
        -> HistoryCNNBranch
        -> history_code [B, branch_dim]

weather: 未来天气序列 [B, 16, 4]
        -> ForecastWeatherMLPBranch
        -> weather_code [B, branch_dim]

direct: 预测点前一刻功率 [B, 1]
        -> direct_code [B, 1]

concat(history_code, weather_code, direct_code)
        -> BayesianLinear
        -> BayesianLinear
        -> mean_head / log_var_head
        -> mean [B, 16], log_var [B, 16]
```

输入和输出：

| 名称 | Shape | 作用 |
| --- | --- | --- |
| `history` | `[batch, 16, 1]` | 预测点前 4 小时历史 `AC_POWER` |
| `weather` | `[batch, 16, 4]` | 未来 16 个步长的天气和小时特征 |
| `direct` | `[batch, 1]` | 预测点前一刻 `last_ac_power`，不使用预测点之后的信息 |
| `mean` | `[batch, 16]` | 未来 16 步预测均值 |
| `log_var` | `[batch, 16]` | 未来 16 步对数方差，代码中会 clamp 到 `[-10, 6]` 避免数值不稳定 |

三路输入各自承担的角色：

| 分支 | 看什么 | 做什么 |
| --- | --- | --- |
| history CNN | 过去 4 小时 `AC_POWER` 曲线 | 用一维卷积提取局部波动、爬坡和下降趋势 |
| weather MLP | 未来 4 小时 `IRRADIATION`、温度和 `hour` | 学习天气预报窗口和未来功率序列之间的非线性关系 |
| direct input | `t-1` 时刻 `AC_POWER` | 提供最近功率水平作为强参考，不经过额外编码 |
| Bayesian fusion | 上面三路拼接后的向量 | 用贝叶斯线性层建模参数不确定性，同时输出均值和方差 |

`BayesianLinear` 把每个权重和偏置都建模为高斯分布，学习后验参数 `mu` 和 `rho`：

```text
sigma = softplus(rho)
weight = mu + sigma * eps
```

损失函数采用近似 ELBO：

```text
loss = Gaussian NLL(mean, log_var, target) + beta * KL / num_batches
```

推理时会执行多次 Monte Carlo 前向传播。多次权重采样造成的预测差异表示模型不确定性；`log_var_head` 输出的方差表示数据噪声不确定性；两者合并后得到最终标准差和预测区间。

## 测试

运行全部测试：

```bash
pytest -q
```

测试覆盖数据处理、时间切分、泄漏检查、指标计算、模型前向传播、训练运行时配置、调参工具和可视化服务等模块。

## 论文写作说明

论文中建议按以下方式描述本实现：

1. 使用过去 4 小时 `AC_POWER` 作为历史出力序列。
2. 使用未来 4 小时天气序列作为条件输入，字段包括辐照度、环境温度、组件温度和小时。
3. 将预测点前一刻 `AC_POWER` 作为直接输入，强调其不是预测点之后的数据。
4. 三路输入融合后进入贝叶斯全连接层，同时输出预测均值和方差。
5. 通过 Monte Carlo 前向传播估计模型不确定性，并结合输出方差形成预测区间。
6. 当前天气输入由真实观测模拟 NWP，后续接入真实数值天气预报后可进一步验证部署性能。

## 当前限制

1. `weather` 使用预测窗口内真实气象观测替代 NWP，实验结论应表述为数据集限制下的模拟预测设定。
2. 当前实验主要基于 Plant 1 单电站数据，跨季节和跨电站泛化能力还需要更多数据验证。
3. baseline 已接入统一训练流程，但正式报告仍建议补充多随机种子结果和消融实验。
