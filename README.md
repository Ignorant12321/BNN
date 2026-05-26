# 光伏功率概率预测实验框架

本项目用于做光伏功率预测实验。当前流程分为三步：先把原始 CSV 预处理并按时间切分，再按单个 YAML 配置训练一个模型，最后加载已训练产物做统一评估和对比。

训练阶段只使用 `train` 拟合模型，并按配置中的 early stopping 监控指标回滚最佳 epoch 和执行早停；训练结束后会额外在 `test` split 上输出最终指标、预测 CSV 和固定时段预测曲线。预测曲线会同时展示均值、90% 预测区间和 95% 预测区间。Torch 后端会只用 train split 拟合标准化参数，在标准化空间训练模型，并在指标、预测 CSV 和图表中反变换回原始 `AC_POWER` 量纲。

光伏功率具有明显昼夜周期，夜间功率长期接近 0。为避免夜间样本使区间覆盖率指标偏高，训练和对比评估会在全时段指标之外，额外输出有效发电时段指标，例如 `train_generation`、`val_generation` 和 `test_generation`。有效发电时段按目标预测时刻的时刻部分定义为 `06:00 <= target_time <= 18:00`。全时段指标反映模型完整运行周期下的整体表现，有效发电时段指标反映模型在实际发电阶段的预测性能。

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
      pv_usibnn.yaml        # 4h 超短期 PV IBNN，使用未来 horizon 外生序列
      0h.yaml               # 不使用 history 序列，只使用未来天气 + 起点前一时刻功率
      1h.yaml               # 只覆盖 lookback=4
      4h.yaml               # 只覆盖 lookback=16
      8h.yaml               # 只覆盖 lookback=32
      12h.yaml              # 只覆盖 lookback=48
      24h.yaml              # 使用默认 lookback=96
    mlp/
      mlp.yaml              # mlp_baseline 默认配置
      24h.yaml
      plain_4h.yaml         # 普通确定性 MLP，4h 超短期对比配置
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

如果配置为 `lookback: 0` 且 `history: []`，模型不会使用历史序列；窗口仍会保留 `direct`，因此可以只使用未来天气和预测起点前一时刻的功率。

未来气象输入中的 `IRRADIATION` 已直接反映太阳辐照条件，并隐含昼夜变化信息。当前默认模型配置不额外加入显式时间编码，以避免和辐照强度信息重复并增加模型复杂度。代码仍支持按实验配置加入由 `DATE_TIME` 自动派生的时间特征。

常用 lookback 配置：

```text
0h  = 0
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

可以给训练目录写入备注，内容会保存到本次时间戳目录下的 `note.txt`：

```powershell
python -m src.experiments.train --config configs/models/bnn/24h.yaml --note "BNN 24h baseline"
```

未传 `--note` 时，`note.txt` 默认写入时间戳目录名。

训练不同 lookback 的 improved_bnn：

```powershell
python -m src.experiments.train --config configs/models/bnn/0h.yaml
python -m src.experiments.train --config configs/models/bnn/1h.yaml
python -m src.experiments.train --config configs/models/bnn/4h.yaml
python -m src.experiments.train --config configs/models/bnn/8h.yaml
python -m src.experiments.train --config configs/models/bnn/12h.yaml
python -m src.experiments.train --config configs/models/bnn/24h.yaml
```

训练 4h 超短期 `pv_usibnn`：

```powershell
python -m src.experiments.train --config configs/models/bnn/pv_usibnn.yaml
```

训练 24h baseline：

```powershell
python -m src.experiments.train --config configs/models/mlp/24h.yaml
python -m src.experiments.train --config configs/models/mlp/plain_4h.yaml
python -m src.experiments.train --config configs/models/cnn/24h.yaml
python -m src.experiments.train --config configs/models/mc_dropout/24h.yaml
```

训练产物目录：

```text
outputs/train/<model_name>/<timestamp>/
  config.yaml
  manifest.json
  note.txt
  train.log
  epoch_history.csv
  metrics.csv
  predictions/test.csv
  figures/loss_curve.png
  figures/prediction_0800_1200.png       # actual / mean / 90% interval / 95% interval
  figures/prediction_1000_1400.png
  figures/prediction_1200_1600.png
  figures/prediction_window_metrics.csv
  models/best.pt             # torch 后端，按配置的验证监控指标回滚后的最佳 epoch
  models/best.pkl            # numpy 后端
```

`metrics.csv` 包含 `train`、`train_generation`、`val`、`val_generation` 和最终 `test`、`test_generation` 指标。训练和早停只使用 `train/val`，`test` 只在模型选定后输出。指标包括：

```text
mae, rmse, nmae, nrmse, picp_90, pinaw_90, picp_95, pinaw_95
```

其中不带 `_generation` 的 split 为全时段指标，带 `_generation` 的 split 为目标预测时刻处于 `06:00-18:00` 的有效发电时段指标。

## 模型配置

模型通过 `model.name` 从注册表构造。当前支持：

```text
improved_bnn
pv_usibnn
mlp_baseline
cnn_baseline
mc_dropout
```

当前 `configs/models/*/*.yaml` 都使用 torch 后端。各模型在 torch 后端是独立结构，不再共用同一个网络：

```text
improved_bnn : 改进 BNN，包含 BayesianLinear、BayesianConv1d 和 KL loss
pv_usibnn    : 超短期 PV IBNN，使用历史功率、当前功率、未来 horizon 辐照/温度/时间序列
mlp_baseline : 普通确定性 MLP，展平 history + future weather/time 后直接输出未来功率
cnn_baseline : 真正 Conv1D baseline，只使用 history 序列
mc_dropout   : MC Dropout，使用 history + future weather，预测阶段多次 dropout sampling
```

`mlp_baseline` 的 torch 版本是普通点预测 MLP，不包含贝叶斯层、卷积层、dropout 采样或方差输出。训练损失使用 MSE；预测 CSV 仍保留 `log_var` 列以兼容统一评估格式，但确定性 MLP 会写入 NaN，因此区间指标 `picp_*` 和 `pinaw_*` 也会是 NaN。

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

`pv_usibnn` 面向 15 分钟粒度的 4h 超短期预测，默认 `lookback: 16`、`horizon: 16`。它保留论文中多分支 IBNN 的思想，但按当前 Plant_1 数据集可用字段落地：

```text
history_power:
  AC_POWER[t-lookback+1 : t]
  -> 1D BayesianConv 分支，kernel=3
  -> AdaptiveAvgPool1d(4)
  -> Flatten
  -> 概率全连接层 32

direct_power:
  AC_POWER[t]
  -> 直接进入合并层

future_solar + future_time:
  IRRADIATION[t+1 : t+horizon]
  hour_sin/hour_cos/dayofyear_sin/dayofyear_cos/is_generation_time[t+1 : t+horizon]
  -> 概率全连接层 32
  -> 概率全连接层 64
  -> 概率全连接层 16

future_weather:
  AMBIENT_TEMPERATURE/MODULE_TEMPERATURE[t+1 : t+horizon]
  -> 概率全连接层 32
  -> 概率全连接层 16

合并后:
  -> 概率全连接层 32
  -> 概率全连接层 16
  -> 输出层 mean / log_var
```

`pv_usibnn` 不给历史功率额外增加 FC 分支，历史出力只走 1D 概率卷积分支，以贴近论文表 3；但针对 `lookback: 16` 的 4h 超短期窗口，CNN 使用 `kernel=3`，并池化到 4 个位置后再展平，避免把 16 个历史点压成单个全局平均值。`IRRADIATION` 不放入温度天气分支，而是作为未来太阳辐照序列和时间特征一起编码。当前数据集没有天气类型、气压、湿度、风速、云量、雾霾、能见度等字段，因此模型不会伪造这些输入，也不做 t-SNE。训练损失仍是 Gaussian NLL 加 `kl_beta * model.kl_loss()`；预测阶段因为 `stochastic_predict=True`，会按 `evaluation.n_samples` 做 MC forward，并合并 epistemic 与 aleatoric 方差。

除 `pv_usibnn` 外，如果把 `training.backend` 改成 `numpy`，注册表会构造轻量 ridge regression 版本，用于测试或无 PyTorch 场景。`pv_usibnn` 使用 BayesianConv1d 和 MC sampling，要求 torch 后端。

常调字段：

```yaml
model:
  hidden_dims: [128, 64] # mlp_baseline 使用
  hidden_dim: 128        # cnn_baseline、mc_dropout 使用
  branch_dim: 64     # cnn_baseline、mc_dropout 使用
  dropout: 0.2       # 仅 mc_dropout 使用

training:
  epochs: 50
  batch_size: 64
  lr: 0.0005
  kl_beta: 0.000001  # improved_bnn 使用
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.0
    metric: val_rmse

evaluation:
  n_samples: 30      # improved_bnn 和 mc_dropout 训练期验证/对比评估的预测采样次数
```

早停只在有验证集时生效，默认监控 `val_rmse`；也可以在 `training.early_stopping.metric` 中指定其他验证指标，例如 4h BNN 使用 `val_generation_nrmse`。`patience: 10` 表示连续 10 个 epoch 没有超过 `min_delta` 的改善就提前结束，并回滚到监控指标最低的权重。

`epochs` 建议理解为训练上限，不建议先跑一次很大的 epoch 后把 early stopping 找到的轮数固定写死。调参阶段通常用较省时的上限，例如 `epochs: 100` 或 `150`，配合 `patience: 10` 或 `15`；最终正式训练可以放宽到 `epochs: 200` 或 `300`，配合 `patience: 15` 或 `20`。如果想诊断模型收敛速度，可以先用较大的上限跑一次训练，再查看 `epoch_history.csv` 中监控指标大概在哪一轮最好，但最终仍建议保留 early stopping。

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

也可以给调参目录写入备注：

```powershell
python -m src.experiments.tune --config configs/tune/bnn.yaml --note "BNN hyperparameter search"
```

默认配置在 `configs/tune/bnn.yaml`，会基于 `configs/models/bnn/24h.yaml` 搜索超参数，并用验证集 `val_rmse` 作为目标指标。默认搜索：

```text
lr, kl_beta
```

BNN 默认不启用 `weight_decay`，网络结构当前按表 3 固定，`batch_size` 统一固定为 64，因此 Optuna 只搜索 `lr` 和 `kl_beta`，不再搜索 `hidden_dim`、`branch_dim`、`conv_kernel` 或 `batch_size`。

4h BNN 可以使用独立调参配置，输出到独立 study，避免和 24h 调参混在一起：

```powershell
python -m src.experiments.tune --config configs/tune/bnn_4h.yaml --note "BNN 4h hyperparameter search"
```

4h 调参完成后，最优参数会写在独立目录中：

```text
outputs/tuning/bnn_4h_generation_optuna/
  tuning_config.yaml
  best_config.yaml
  trials.csv
  runs/trial-0000/
```

Optuna 只负责生成超参数组合；每个 trial 仍然调用项目现有训练流程。只要训练配置是 `training.backend: torch`，训练器就会使用 `torch.optim.AdamW`，因此 BNN 调参时实际是：

```text
Optuna 选参数 -> AdamW 训练 -> 读取配置中的 metric -> Optuna 更新搜索
```

默认 BNN 调参输出是稳定目录，不带时间戳：

```text
outputs/tuning/bnn_optuna/
  note.txt
  tuning_config.yaml
  study.db
  trials.csv
  best_config.yaml
  best_run.txt
  runs/trial-0000/
  runs/trial-0001/
```

`study.db` 是 Optuna 的 SQLite storage，用于断点继续。`n_trials` 表示目标完成 trial 总数：如果中断后只完成了 7/20，再运行同一个命令会继续补到 20；如果已经完成 20/20，再运行不会重复训练。需要继续扩展搜索时，把对应 tune YAML 里的 `n_trials` 改大即可，例如把 `configs/tune/bnn_4h.yaml` 从 `50` 改成 `80` 后会继续从 50 补到 80。

一般同一个模型和 lookback 只需要完成一组 study。想从 0 开始重新跑同一个调参配置，可以删除对应调参目录，例如：

```powershell
Remove-Item -Recurse -Force outputs\tuning\bnn_4h_generation_optuna
```

然后重新运行 `python -m src.experiments.tune --config configs/tune/bnn_4h.yaml`。4h 调参默认使用 `val_generation_nrmse` 作为目标指标。如果想保留旧结果并新开一组实验，更推荐改 tune YAML 里的 `name` 和 `study_name`，例如 `bnn_4h_generation_optuna_v2`，这样会输出到新的稳定目录。删除 tuning 目录只会删除 Optuna study 和 trial 产物，不会撤销已经写入 `configs/models/bnn/4h.yaml` 的参数；如果要完全从公共默认配置重新搜索，需要先移除 4h 配置中已应用的 `training.lr` 和 `training.kl_beta` 覆盖。

调参 trial 的训练产物不会写到顶层 `outputs/train/`，而是写入当前 tuning 目录下的 `runs/trial-xxxx/`。

应用最优参数前可以先预览变化。默认/24h 调参通常写回 BNN 公共配置：

```powershell
python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/bnn_optuna --target configs/models/bnn/bnn.yaml
```

4h 独立调参建议只写回 4h 配置，避免影响其他 lookback：

```powershell
python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/bnn_4h_generation_optuna --target configs/models/bnn/4h.yaml
```

命令会展示类似：

```text
Tuned Parameter Changes
------------------------------------------------
training.lr      | 0.0005 -> 0.0009291350888559107
training.kl_beta | 1e-06 -> 4.182709268632557e-05
```

然后提示是否应用。确认后才会写回目标 YAML。若确认无误，也可以跳过提示：

```powershell
python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/bnn_4h_generation_optuna --target configs/models/bnn/4h.yaml --yes
```

注意：`configs/models/bnn/0h.yaml`、`1h.yaml`、`4h.yaml`、`8h.yaml`、`12h.yaml` 和 `24h.yaml` 都会 include 公共 `bnn.yaml`。如果把 4h 搜到的参数写入 `bnn.yaml`，其他 lookback 后续训练也会继承这组 4h 参数；只想更新 4h 时应写入 `configs/models/bnn/4h.yaml`。

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

对比输出也支持备注：

```powershell
python -m src.experiments.compare --config configs/compare/main.yaml --note "24h main comparison"
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
  note.txt
  model_metrics.csv
  predictions/<label>.csv
  figures/loss_curves.png
  figures/prediction_0800_1200.png
  figures/prediction_1000_1400.png
  figures/prediction_1200_1600.png
```

## 本地可视化对比

启动本地 visualizer：

```powershell
python visualizer/server.py
```

浏览器打开终端输出的地址，默认是 `http://127.0.0.1:5177/`。页面默认为空，只加载可选文件夹列表，不会自动把所有结果铺到表格里。

- 读取：选择 `outputs/comparisons/<timestamp>`，读取其中的 `model_metrics.csv`、`predictions/`、配置和关键图像进行展示。
- 创建：勾选多个 `outputs/train/<model>/<timestamp>`，点击“创建并读取”，固定用 `test` split 生成新的 `outputs/comparisons/<timestamp>` 后再展示。
- 导航：筛选和显示当前已读取 comparison 中的 run。
- 对比图像只保留关键图：`loss_curves.png` 和 `prediction_*.png`。
- 清空页面：只清空当前浏览器展示，不删除文件。
- 退出当前文件夹：退出当前已读取的 comparison，让页面回到空表状态，不删除任何文件。

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
3. 用 `outputs/tuning/bnn_optuna/best_config.yaml` 重训或作为参考更新 BNN 默认配置；如果调的是 4h，则用 `outputs/tuning/bnn_4h_generation_optuna/best_config.yaml` 更新 `configs/models/bnn/4h.yaml`。
4. 固定候选超参数后，训练 `improved_bnn` 和三个 baseline。
5. 用 `configs/compare/main.yaml` 做 24h 主模型和 baseline 对比。
6. 分别训练 `0h/1h/4h/8h/12h/24h` 的 `improved_bnn`。
7. 用 `configs/compare/lookback.yaml` 做不同历史窗口对比。
8. 只用 `train/val` 调参，最后再用 `test` 结果写最终对比。

## 测试

```powershell
pytest -q
```
