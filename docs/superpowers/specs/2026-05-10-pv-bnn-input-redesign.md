# PV BNN Input Redesign

## Goal

重构光伏概率预测模型的输入定义，使代码、配置和 README 与用户给定的论文结构一致，而不是继续沿用现有 README 中错误的四分支解释。

## Correct Input Semantics

- `history` 是历史光伏出力序列，只包含预测点之前 4 小时的 `AC_POWER`。在 15 分钟粒度下，`lookback=16`，窗口为 `[t-16, ..., t-1]`。
- `weather` 是预测窗口天气序列，表示可由数值天气预报得到的未来天气。当前数据集中没有真实 NWP，因此实验用目标窗口真实值模拟可获得天气预报，字段为 `IRRADIATION`、`AMBIENT_TEMPERATURE`、`MODULE_TEMPERATURE` 和数值小时 `hour`。
- `direct` 是预测点前一刻的 `AC_POWER`，只取 `t-1`，不能取预测点 `t` 或预测点之后的数据。
- `target` 是未来 4 小时 `AC_POWER` 序列，窗口为 `[t, ..., t+15]`。

## Architecture

模型保留三路输入和概率输出：

- 第一部分：`weather` 序列展平后进入全连接分支，结构为 `Linear(4 * horizon, 32) -> Linear(32, 64) -> Linear(64, branch_dim)`。
- 第二部分：`history` 序列进入 1D-CNN 分支，两层卷积和池化后输出定长表示。
- 第三部分：`direct` 标量作为预测点前一刻输入，直接拼接进融合层。
- 三路表示合并后进入贝叶斯全连接融合层，并输出未来 16 步预测均值和对数方差。

## Data Flow

`make_window_arrays` 按时间排序后构造窗口：

- `history = rows[start:hist_end][["AC_POWER"]]`
- `weather = rows[hist_end:target_end][weather columns]`
- `direct = rows[hist_end - 1]["AC_POWER"]`
- `target = rows[hist_end:target_end]["AC_POWER"]`

切分仍然先按时间划分 train/val/test，再在各子集内部构造窗口，避免窗口跨越数据边界。Scaler 仍然只在训练集拟合。

## Testing

新增或更新测试覆盖：

- 特征分组只包含新语义需要的列。
- `lookback=16` 配置在默认、调参和对比配置中一致。
- 窗口构造证明 `history` 只含过去 AC_POWER，`weather` 用目标窗口天气真实值，`direct` 只取 `t-1` 的 AC_POWER。
- 模型 forward 接受三路输入并输出 `[batch, horizon]` 的均值和方差。

## Documentation

README 删除旧的“历史出力和历史气象”“direct 三变量”“单独 time 分支”“可关闭未来天气”的描述，改为说明当前实验把未来实测天气作为 NWP 替代输入，并明确这是一种数据集限制下的模拟设定。
