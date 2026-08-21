# src/ml — 机器学习资产定价

复现 Gu, Kelly & Xiu (2020) *Empirical Asset Pricing via Machine Learning*
(RFS 33:5, 2223-2273) 的框架，用于 A 股。

---

## 运行

```zsh
cd /Users/louis/MyProjects/Quant
conda activate /Users/louis/MyProjects/venv

python -m src.ml.run --dry-run                 # 只看面板与切分方案
python -m src.ml.run --fast                    # 缩小网格（约 15 分钟）
python -m src.ml.run --full                    # 完整网格（约 2.5 小时）
python -m src.ml.run --models EW-Sign,LGBM     # 只跑指定模型
python -m src.ml.run --features pool           # 只用合成池那 10 个特征
```

**前置条件**：`output/cache/` 里要有因子缓存，即先跑过 `python -m src.main`。
本模块**只读缓存、不重算因子**。所有默认值在 `run.py` 顶部的 `ML_CONFIG`。

### 算力预算：不再默认吃满所有核

原先 RF / LGBM 写死 `n_jobs=-1`，HistGBR 与 BLAS 也默认开满逻辑核，
在 MacBook 上连跑几十分钟会烫手，还会热降频（P 核降频后单窗耗时不降反升）。
现在统一由 `src/config/compute.py` 管：

| 方式 | 效果 |
|---|---|
| 默认 | 逻辑核的一半（本机 18 核 → **9 线程**，约慢 1.85 倍） |
| `--jobs 4` | 本次只用 4 线程 |
| `--cooldown 10` | 每个滚动窗口之间停 10 秒散热（**不影响结果**，窗口间本就无状态） |
| `QUANT_JOBS=4` | 环境变量，对 `src.main` 等所有入口生效 |
| `--jobs all` / `QUANT_JOBS=all` | 恢复「吃满」的旧行为，最快也最热 |

线程数与散热间歇是两件事：前者压**峰值功耗**，后者压**持续积热**，可叠加。
线程预算同时作用于 sklearn 的 `n_jobs`、OpenMP（HistGBR / LightGBM）和
numpy 背后的 BLAS——环境变量那一层由 `src/__init__.py` 在 numpy 导入前设好，
运行时再由 threadpoolctl 兜底。

**代价**（RandomForest 300 树，20 万 × 10 特征，本机实测）：

| 线程数 | 18 | 12 | 9（默认） | 6 |
|---|---|---|---|---|
| 耗时 | 8.0s | 11.0s | 14.8s | 22.1s |

近似线性——异构核没有惩罚（树之间无同步屏障，E 核照样出力）。
所以默认的一半线程约合 **1.85 倍时长**换约一半功耗。

⚠️ 改线程数**不改变任何结果**：模型都固定了 `random_state`，上面四组的
`predict` 输出逐位相同（-0.016187 / -0.015971 / …）。并行度只影响速度。

输出在 `output/ml/`：`model_summary.csv`、`benchmark_compare.csv`、
`dm_matrix.csv`、`variable_importance.csv`、`params_<模型>.csv`。

---

## 设计的核心：只做预测，不碰既有回测链路

本模块的产出是一张「月末 × 股票」的预测收益宽表 ŷ，直接喂给既有组件：

```python
pred_wide = to_wide(res["preds"]["LGBM"])
grp = group_return(pred_wide, fwd_ret, n_groups=10)   # backtest/engine.py
ic  = calc_ic(pred_wide, fwd_ret, method="spearman")  # backtest/metrics.py
```

因此 ML 与既有 62 个单因子走**完全相同**的评估口径，天然可比，
且不需要改动任何既有代码。既有五分组结果与文档记录全部保持有效。

---

## 模块

| 文件 | 职责 |
|------|------|
| `dataset.py` | 宽表因子 → MultiIndex(date, code) 长面板；截面 rank→[-1,1]；标签对齐 |
| `cv.py` | 滚动窗口 60/24/12，每个窗口跑泄漏断言 |
| `models.py` | 模型库 + 验证集调参（Huber 损失） |
| `evaluate.py` | R²_oos / Diebold-Mariano / 置零法重要性 / 张成检验 |
| `benchmark.py` | 同口径基准 + 换手率 + 交易成本敏感性 |
| `pipeline.py` | 滚动训练 → 样本外预测 → 接回既有回测 |
| `run.py` | 入口 + `ML_CONFIG` |

---

## 三个必须理解的口径

### 1. 标签是**原始超额收益**，不做秩变换

特征做 rank→[-1,1]，但**标签不做**。因为 R²_oos 的定义是

```
R²_oos = 1 − Σ(y − ŷ)² / Σy²          ← 分母不去均值，以 0 为基准
```

一旦标签也做秩变换，这个比值就失去「相对于预测 0」的经济含义，
算出来的数字无法与 GKX 的 0.3%~0.4% 对照。

分母用 0 而非历史均值，是 GKX 的刻意选择：个股月度收益的历史均值是极度
嘈杂的估计量，用它当基准会人为抬高 R²。所以 R²_oos > 0 的含义是
**「比闭着眼睛说所有股票超额收益都是 0 要好」**——一条比听起来低得多的及格线。

**0.4% 就是这个领域的天花板级结果。** 不要拿它和别处见过的 R²=0.3 比。

### 2. 调参用 **Huber 损失**，不是 MSE

这是 GKX 的核心手法，不是可选项。本项目面板 717,431 个「股票-月」观测里，
有 45 个单月收益 >200%，最大 **+1073%**（688585.SH 2025-07）。
用 MSE 选超参，这几十个点会主导整个验证集损失，选出来的参数是在拟合极端值。

**标签不做缩尾**（德国那个复现项目做了，作者自己承认是对 GKX 的偏离）——
缩尾会同时改变 R²_oos 的分子分母，破坏可比性。极端值交给 Huber 处理。

### 3. 分组数 **10**，与既有 5 分组**不可直接比**

十分档多空端更极端，LS 收益天然更高，**这是机械效应不是能力差异**。
`benchmark.py` 会把既有因子也按 10 组、在 ML 的样本外区间重算，
四个条件（区间/分组/加权/股票池）全部对齐后再比。

---

## `EW-Sign`：最关键的对照组

模型列表第一个是 `EW-Sign`（滚动符号等权），它回答一个决定性的问题：
**ML 的增量到底来自「特征更宽」还是「权重学出来的」？**

已知两件事同时成立：

- 在合成池那 10 个特征上，学习**打不过**等权（夏普 1.0 vs 1.19）
- 换到 61 个特征，ML 夏普跳到 2.0~2.3

这有两种互斥解释：

| | 含义 |
|---|---|
| (a) 宽度本身就够 | `EW-Sign` 用 61 特征也能到 2.0 → 模型没贡献，价值在特征工程 |
| (b) 学习需要宽度才发挥 | `EW-Sign` 只能到 1.4 → 模型确实有贡献 |

`EW-Sign` 就是 (a) 的实现：同样 61 个特征、同样滚动窗口，但权重恒等权。
**它与其他模型的差值 = 学习的净贡献。**

方向必须**滚动决定**：既有 `main.py` 的 `FACTOR_DIRECTIONS` 是按全样本 IC
人工填的，直接用就是前视偏差。这里只用训练段的 IC 符号，与 ML 看到的信息一致。

它还会拟合一个一元映射 `ŷ = a + b·score` 把得分放到收益量纲上——
只有 2 个参数、**不改变任何排序**（秩相关恒为 1），因此仍是「等权」，
只是变成了可比的收益预测（否则 R²_oos 会因尺度不对而无意义）。

---

## 模型清单

| 模型 | 说明 |
|------|------|
| `EW-Sign` | ★ 对照组，见上 |
| `OLS-H` | 全特征线性 + Huber；无超参，train+val 合并训练 |
| `OLS-3` | GKX 的简单基准。原文用 size/bm/mom，**A股动量全期限失效**，故以 `reversal_20` 代替 |
| `ENet` | 弹性网 |
| `PLS` / `PCR` | 偏最小二乘 / 主成分回归 |
| `RF` | 随机森林 |
| `GBRT` | 梯度提升树 |
| `LGBM` | LightGBM（可选依赖） |

### ⚠️ GBRT 用的是 `HistGradientBoostingRegressor`，不是 `GradientBoostingRegressor`

GKX 与德国项目用后者（精确分裂，支持 `loss='huber'`）。在他们 214,016 行的
面板上尚可，但本项目单个训练窗有 231,302 行 × 61 特征，实测
**仅 2 组超参、单个窗口就超过 10 分钟**（单线程逐样本精确分裂），
13 个窗口要 2 小时以上，不可行。

换成直方图实现后：**7.2 秒**（约 100 倍）。代价是它不支持 Huber，
可选损失只有 squared / absolute / poisson / quantile，故取
**absolute_error（L1）**——Huber 本就是 L2 与 L1 的分段拼接，δ 之外那段正是 L1，
稳健性同源。真正忠于 GKX 的 Huber 目标由 **LGBM**（`objective="huber"`）提供，
两者并列互为对照。

### 超参网格按**本项目样本量**标定

不能照抄德国那份：他们 214,016 个观测、GBRT 网格用
`min_samples_split ∈ {5000, 8000, 10000}`；本项目 716,052 行（3.3 倍），
照抄会严重欠拟合。

---

## 安装 LightGBM（可选）

```zsh
conda activate /Users/louis/MyProjects/venv
conda install -c conda-forge lightgbm
```

用 conda-forge 而非 pip：macOS ARM 上 LightGBM 依赖 `libomp`，
conda-forge 会自动带上；pip 的 wheel 在部分机器上会因找不到 `libomp.dylib`
而 import 失败。未安装时 `has_lightgbm()` 会自动跳过该模型。

---

## 数据规模

| | 值 |
|---|---|
| 面板 | 716,052 行 × 61 特征 |
| 区间 | 2007-01 ~ 2026-07（234 个月） |
| 股票 | 5,793 只（时点并集，含退市） |
| 滚动窗口 | 13 个（60/24/12） |
| **样本外** | **2014-01 ~ 2026-07，151 个月** |

`seasonality` 因缺失率 88% 被自动剔除（需要 6~10 年同月历史）。
样本量是德国复现项目 214,016 行的 **3.3 倍**。
