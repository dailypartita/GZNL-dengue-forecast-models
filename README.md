# GZNL-dengue-forecast-models

广州国家实验室（GZNL Group）的中国登革热月度发病数预测模型，面向
[China Infectious Disease Forecast Hub](https://github.com/dailypartita/China-Infectious-Disease-Forecast-Hub)
的目标 **`mo inc dengue cases`**。

方法改编自 EpiETL `modeling` 套件（`backtest.py`、`ensemble_conformal.py`、
`forecast_final.py`、`prepare_data.py`）。原套件使用本地全国序列和 5 个分位数；
本仓库改为只使用枢纽官方月度全国登革热报告病例数，输出枢纽要求的 23 个分位数，
并按提交日历把模型步长 1–5 个月对应到 hub horizon −1…3。地点仅为 `CN`，单位是
非负整数病例数。

本目录提交的模型是 **GZNL-DengueLGBMCQR**（LightGBM 分位数回归 + 保形校准）。
`designated_model` 设为 `false`：GZNL 在该 hub 已有两个指定模型（NextWave、
VincentAvg），一队最多指定两个。模型简称带有 `Dengue` 前缀，避免与其他病种的同名方法冲突。

该模型已注册并提交 2026-11-01 轮预测，见枢纽仓库中的
[`GZNL-DengueLGBMCQR`](https://github.com/dailypartita/China-Infectious-Disease-Forecast-Hub/tree/main/model-output/GZNL-DengueLGBMCQR)。

## 预测目标

| 字段 | 取值 |
|------|------|
| `target` | `mo inc dengue cases` |
| `location` | `CN` |
| 单位 | 全国月度报告病例数，非负整数 |
| 提交步长 | 整数 `−1` 到 `3` 个月 |
| 分位数 | hub 规定的 23 个水平（0.01 … 0.99） |
| `reference_date` | 当月 1 日；`target_end_date = reference_date + horizon 个自然月` |

Hub 登革热的 `reference_date` 是当月 1 日。以 2026-11-01 轮为例，最后一次观测是
**2026-09-01**，因此 hub horizon −1、0、1、2、3 分别对应模型 lead 1 到 5
（目标月 2026-10 至 2027-02）。日期一律用日历月偏移，绝不用周。

## 训练集

训练数据来自官方 hub 文件 `target-data/time-series.csv`，筛选
`target == "mo inc dengue cases"` 且 `location == "CN"`。本仓库
`data/time-series.csv` 是生成 2026-11-01 提交时使用的快照。

该快照中，登革热序列一共有 **273 个整月观测**，时间范围是
**2004-01-01 至 2026-09-01**，中间没有缺月，全部为有限非负数值。
历史峰值出现在 2014-10（28,796 例），2024-10 为近年高峰（11,083 例）。
2026 年近期观测：01=93，02=94，03=92，04=137，05=222，06=255，07=1,120，08=836，09=486。

这不是一次固定的“前 70% 训练、后 30% 测试”划分。每个预测起点都会重新拟合，
并且只使用当时已经发布的观测。特征需要 12 个月滞后才能构成完整设计矩阵
（序列从第 12 个点，也就是 2004-12，开始进入训练）。在 2026-11-01 这次提交中，
各步长的训练样本数为：

| hub horizon | 从最后观测算起的步数 k | 训练样本数 |
|-------------|-------------------------|------------|
| −1 | 1 | 261 |
| 0 | 2 | 260 |
| 1 | 3 | 259 |
| 2 | 4 | 258 |
| 3 | 5 | 257 |

回测第一个评估起点 2023-09（当时可见到 2023-09）的训练样本约为 225–229 行，
随后随历史拉长逐渐增加到 250 行以上。不用气候、ENSO 或其他外部协变量：
本地 EpiETL 协变量更新节奏与枢纽序列并不一致，提交模型只依赖这条官方病例序列。

## 算法原理

提交模型 **DengueLGBMCQR** 分两步：先做直接多步 LightGBM 分位数回归，再用滑动窗口
[CQR](https://arxiv.org/abs/1905.03222)（Romano, Patterson, Candès, 2019）校准区间。

**1. LightGBM 分位数回归。** 对每个预测步长、每个分位数水平单独训练一个模型
（直接多步，而不是迭代滚动一步预测）。损失函数是分位数（pinball）损失，
水平 `α` 取 hub 规定的 23 个分位数。树模型超参数为：`n_estimators=300`，
`learning_rate=0.05`，`num_leaves=15`，`min_child_samples=5`。训练在 `log1p`
空间进行，预测后再 `expm1` 并截断到 ≥ 0。

特征在最后一次可见观测处构造，不使用未来信息：

- 滞后 1、2、3、6、12 个月（`lag_k` 为最近第 k 个观测，故 `lag1 = y_t`，`lag12 = y_{t-11}`）
- 近 3 个月与近 12 个月滚动均值（`r3m`、`r12m`）
- 一阶差分（`d1 = lag1 − lag2`）
- 目标月的年周期 Fourier 项（月份的 1 阶正弦和余弦）

拟合后的 23 个分位数会做保序处理（`np.maximum.accumulate`），避免分位数交叉。

**2. 保形分位数回归（CQR）。** 对每个对称分位数对 `(τ, 1−τ)`（`τ < 0.5`），用最近最多 24 个
已实现起点的预测，计算一致性得分 `S = max(q_lo − y, y − q_hi)`，再按
`ceil((n+1)(1−α))/n` 分位数（`α = 2τ`，`method="higher"`）把区间**外扩**为
`[q_lo − Q, q_hi + Q]`，下界截到 0，中位数不动。前 10 个起点只输出未经 CQR 的分位数。
CQR 之后再次保序、非负，并四舍五入为整数病例数；下侧分位数不会被重排到超过中位数。

## 如何确保模型可靠

可靠性主要靠「不泄露未来信息」的滚动起点回测，以及区间校准来约束。

- **无前瞻。** 起点月 `t` 只能看到 `target_end_date ≤ t` 的观测；特征、训练标签和 CQR 校准分数都在这条截断线之内。
- **滚动起点回测。** 评估窗口为月度原点 2023-09 至 2026-04（此窗口内 lead 1–5 的结局均已观测），共 32 个起点 × 5 个步长 = 160 对。滚动预测从更早的起点开始，以便 24 个月 CQR 窗口在评估开始前预热。
- **公平比较窗口。** 指标只在评估窗口上计算，此时每个步长的校准样本数已达 20–24，避免把尚未校准的早期点混进来。
- **对照与事先规则。** 同一套回测里同时评估未校准 LightGBM 与 LightGBM+CQR。提交方法按平均加权区间分数（WIS，23 个分位数得分的平均）选择，结果是 LightGBM+CQR。
- **覆盖率检查。** 除 MAE、WIS 外，报告 50%、80%、95% 区间的经验覆盖。CQR 后 50%/80%/95% 覆盖从 0.25/0.46/0.72 提到 0.51/0.91/0.98；50% 接近名义水平，80% 与 95% 偏保守。点预测的 MAE 与未校准 LightGBM 几乎相同（中位数不被 CQR 移动）。
- **输出约束。** 预测值限制为非负整数病例数，并强制分位数单调，减少无效区间。
- **格式校验。** 提交前检查列名、行数、分位数集合、单调性、非负整数和日期关系；hub 仓库的 CI 会在 pull request 上运行 hubValidations。

需要注意的限制：训练使用了 2004 年以来的全部可见历史，没有单独做疫情年份剔除或外部协变量（如气候、蚊媒、输入病例）；CQR 的 24 个月窗口包含 2024-10 高峰（11,083 例），会把 90%/95% 上侧区间拉得很宽，覆盖率因此偏保守。这是方法本身的结果，提交时未再人为收紧尾部。

## 回测摘要

指标在公平评估窗口上计算（32 个起点 × 5 个步长，共 160 个预测点）。WIS 为 23 个分位数得分的平均。

| 方法 | n | MAE | WIS | 50% 覆盖 | 80% 覆盖 | 95% 覆盖 |
|------|---|-----|-----|----------|----------|----------|
| **lightgbm_cqr（提交）** | 160 | 734 | **551** | 0.51 | 0.91 | 0.98 |
| lightgbm | 160 | 734 | 572 | 0.25 | 0.46 | 0.72 |

按 WIS 选择 LightGBM+CQR。CQR 各步长的 WIS 为 406、474、519、659、697。详细分步长、分月份指标见 `analysis/backtest_metrics.csv`。

## 可靠性图示

图由 `scripts/plot_reliability.py` 根据公平回测子集（每步长 32 个起点，共 160 个预测点）生成；`figures/` 中同时提供 PNG 和可编辑的 PDF。

**图 1. 预测精度随步长的变化。** LightGBM+CQR 在各步长的加权区间分数（WIS）均优于未校准 LightGBM。CQR 不移动中位数，因此两条曲线的 MAE 重合，WIS 略有改善。误差随步长增加而上升。

![图 1 各步长的 WIS 与 MAE](figures/fig1_accuracy_by_horizon.png)

**图 2. 区间校准与锐度。** (a) CQR 后 50% 覆盖接近名义水平，80% 与 95% 覆盖从明显偏低转为略保守；(b) 未校准 LightGBM 的概率积分变换（PIT）在 1 附近堆积，说明区间过窄，CQR 后更接近均匀；(c) 代价是 95% 区间明显变宽，50% 区间宽度几乎不变。

![图 2 覆盖率、PIT 与区间宽度](figures/fig2_calibration.png)

**图 3. 回测轨迹。** +1 个月与 +4 个月预测的中位数和区间叠加在观测序列上，能够跟上 2024 年和 2025 年的秋季高峰；更长步长时中位数峰值偏低、区间明显变宽。

![图 3 回测中的预测与观测](figures/fig3_backtest_timeseries.png)

**图 4. 2026-11-01 提交预测。** 在最近一次观测 486 例之后，中位数继续回落，95% 区间随步长加宽。图中同时画出 2024-10 高峰，以便对照 CQR 上侧区间的宽度来源。

![图 4 本轮提交的扇形预测图](figures/fig4_submission_forecast.png)

**图 5. 可靠性图。** 将名义覆盖与经验覆盖对照。未校准 LightGBM 在各水平上均偏低；CQR 后 50% 落在对角线上，更高名义水平略偏保守（区间略宽）。

![图 5 名义覆盖与经验覆盖](figures/fig5_reliability_diagram.png)

## 本次提交（reference_date = 2026-11-01）

最后一次观测是 2026-09-01，报告病例 **486**。中位数与 90% 区间：

| horizon | target_end_date | 5% | 中位数 | 95% |
|---------|-----------------|----|--------|-----|
| −1 | 2026-10-01 | 0 | 298 | 4,089 |
| 0 | 2026-11-01 | 0 | 210 | 2,802 |
| 1 | 2026-12-01 | 0 | 115 | 3,837 |
| 2 | 2027-01-01 | 0 | 61 | 9,062 |
| 3 | 2027-02-01 | 0 | 41 | 8,367 |

2026-11-01 是该目标的合法轮次。按枢纽日历，提交截止为北京时间周三 23:59；本轮首次提交在周四完成，有可能按迟交处理。随后已将模型编号从 `GZNL-LGBMCQR` 更名为 `GZNL-DengueLGBMCQR`，预测数值未改。

## 如何运行

```bash
bash run.sh
```

`run.sh` 会在需要时创建 `.venv`、安装 `requirements.txt`，并运行：

```bash
python -m models.lgbm_cqr
```

这会用 `data/time-series.csv` 训练并写出
`output/GZNL-DengueLGBMCQR/2026-11-01-GZNL-DengueLGBMCQR.csv`
（5 个 horizon × 23 个分位数 = 115 行数据 + 表头，UTF-8 无 BOM，8 列）。

回测（不阻塞提交）：

```bash
source .venv/bin/activate
python -m analysis.backtest
```

默认在 2023-09 至 2026-04 的月度原点上评估，并会更早开始滚动预测以便 CQR 窗口预热。
指标写入 `analysis/backtest_metrics.csv`。

根据已有回测缓存重画本仓库中的图：

```bash
source .venv/bin/activate
python scripts/plot_reliability.py
```

本机未安装 R / hubValidations，提交前做了列名、行数、分位数集合、单调性、
非负整数和日期关系的格式检查。仓库 CI 会在 pull request 上运行 hubValidations。

## 目录

```
models/               枢纽日历、LightGBM 与 CQR
analysis/             滚动起点回测与指标
scripts/plot_reliability.py   本仓库可靠性图
requirements.txt
run.sh                训练并写出提交 CSV 的入口
model-metadata/       模型注册 YAML 副本
output/               hubverse 预测 CSV
data/time-series.csv  本次运行使用的官方 time-series 快照
figures/              回测与提交图（PNG 与 PDF）
```
