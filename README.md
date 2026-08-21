# Quant-CN-markets — A 股量化因子回测框架

模块化因子研究平台，支持因子构建、股票池过滤、分组回测、IC 检验与净值可视化。
数据源为 Database（主）+ `_archive/raw_data`（存档补充）。

---

## 快速开始

### 运行回测

```bash
# 激活共享环境（上层目录 venv/，实为 conda 环境，Python 3.13）
conda activate /Users/louis/MyProjects/venv

# 运行回测
# 注意：以包方式运行，须在 Quant/ 根目录执行
cd /Users/louis/MyProjects/Quant
python -m src.main
```

代码以 `src/` 包形式组织，内部统一使用 `from src.xxx import ...` 绝对导入。
必须用 `python -m src.main` 运行（不能 `python src/main.py`）。

**输出位置**：
- 回测结果 CSV：`output/A/stats/` 和 `output/HK/stats/`
- 净值曲线图：`output/A/img/` 和 `output/HK/img/`
- 因子缓存：`output/cache/`（命中缓存则跳过计算）

**依赖**：见 `requirements.txt`

### FF3 因子构建（`ivol` / `ff3_betas` 依赖）

数据更新后需重建 FF3，再跑回测：

```bash
# 查看 FF3 状态
python -m src.ff3_builder --dry-run

# 如果需要重建 FF3（数据更新后）
python -m src.ff3_builder
```

### 机器学习资产定价（`src/ml/`）

复现 Gu, Kelly & Xiu (2020) 框架。**前置条件：先跑过 `python -m src.main`**——
ML 模块只读 `output/cache/` 里的因子缓存，自己不重算因子。

```bash
cd /Users/louis/MyProjects/Quant
conda activate /Users/louis/MyProjects/venv

python -m src.ml                          # 按 run.py 顶部的 ML_CONFIG 跑
python -m src.ml --dry-run                # 只看面板与滚动切分方案，不训练
python -m src.ml --fast                   # 缩小超参网格（约 15 分钟）
python -m src.ml --full                   # 完整网格（约 2.5 小时）
python -m src.ml --models EW-Sign,ENet,PLS  # 只跑指定模型（跳过慢的 RF）
python -m src.ml --features pool          # 只用合成池那 10 个因子
python -m src.ml --n-groups 5             # 改成五分组（默认 10，GKX/JKP 惯例）
```

`python -m src.ml` 与 `python -m src.ml.run` 等价（前者只是转发）。

**输出位置**：`output/ml/`

| 文件 | 内容 |
|------|------|
| `model_summary.csv` | ★ 主表：R²_oos（全/大盘/小盘）、IC、ICIR、LS 年化、夏普、回撤、胜率 |
| `benchmark_compare.csv` | ★ 同口径对比：ML vs 既有合成/单因子，**含换手率与 0/15/30bp 成本敏感性** |
| `dm_matrix.csv` | Diebold-Mariano 两两检验（带显著性星号） |
| `spanning_test.csv` | 张成检验：ML 多空收益对既有因子回归，看 α 是否显著 |
| `variable_importance.csv` | 置零法变量重要性 |
| `params_<模型>.csv` | 各模型在 13 个窗口分别选中的超参与验证损失 |

**可选依赖 LightGBM**（不装会自动跳过该模型）：

```bash
conda install -c conda-forge lightgbm     # 用 conda-forge，会自动带上 libomp
```

口径说明、`EW-Sign` 对照组的作用、以及 GBRT 实现上的偏离，见
[`src/ml/README.md`](src/ml/README.md)。

---

## 项目结构

```
Quant/                         # ← git 跟踪：仅 src/ 代码 + 项目文件
├── requirements.txt           # 依赖说明（共用上层 venv，Python 3.13）
├── README.md  CLAUDE.md  .gitignore
├── src/                       # 源码包，入口 python -m src.main
│   ├── __init__.py
│   ├── main.py                # 唯一运行入口
│   ├── convert_report_to_pdf.py
│   ├── config/
│   │   └── settings.py        # 全局路径、因子参数、中性化开关
│   ├── data/
│   │   ├── loader.py          # 统一数据加载（自动拼接存档与 Database）
│   │   └── universe.py        # 股票池过滤（剔除 ST、停牌、次新股）
│   ├── factors/
│   │   ├── base.py            # 预处理流水线（去极值 → 中性化 → 标准化）
│   │   ├── microstructure.py  # 微观结构因子
│   │   ├── fundamental.py     # 基本面因子
│   │   ├── expansion.py       # 特征扩充（动量/波动/流动性/规模族）
│   │   └── FACTORS.md         # 因子构建方法文档
│   ├── backtest/
│   │   ├── engine.py          # 月度分组回测核心
│   │   ├── metrics.py         # 绩效指标（IC、ICIR、Sharpe、最大回撤等）
│   │   └── report.py          # 报告输出（控制台 + CSV + PNG）
│   ├── strategy/
│   │   ├── optimizer.py       # 因子打分与分组工具
│   │   ├── scoring.py         # 因子打分
│   │   └── combine_factors.py # 多因子合成（方向对齐 → 正交化 → ICIR 权重）
│   └── ml/                    # 机器学习资产定价，入口 python -m src.ml
│       ├── __main__.py        # 转发到 run.py
│       ├── run.py             # 入口 + ML_CONFIG
│       ├── dataset.py         # 宽表 → 长面板、截面 rank、标签对齐
│       ├── cv.py              # 滚动窗口 60/24/12 + 泄漏断言
│       ├── models.py          # 模型库 + 验证集调参（Huber 损失）
│       ├── evaluate.py        # R²_oos / DM 检验 / 变量重要性 / 张成检验
│       ├── benchmark.py       # 同口径基准 + 换手率 + 交易成本
│       ├── pipeline.py        # 滚动训练 → 样本外预测 → 接回既有回测
│       └── README.md          # 口径说明与设计决策
│
├── output/                    # 回测输出 CSV + PNG（不跟踪）
│   └── ml/                    # ML 结果（不跟踪）
├── reports/                   # HTML / pptx 报告（不跟踪）
└── _archive/                  # 历史代码 + raw_data + 旧脚本 defense/run_backtest（不跟踪）
```

---

## 主要配置（`main.py`）

### 回测区间

```python
BACKTEST_START = "2007-01-01"   # 起始日期。数据本身 2005-01-04 起可用；
                                #   选 2007 是主动判断——2005~2006 仅 1300~1400 只、
                                #   股改期间大量停牌、估值数据质量差
BACKTEST_END   = <自动取上月末>  # 截止日期（动态，每次运行自动更新）
```

### 因子开关（`FACTOR_FLAGS`）

将需要运行的因子设为 `True`，其余保持 `False`：

```python
FACTOR_FLAGS = {
    "turnover_20":             True,   # 换手率（无中性化）
    "turnover_20_neutral":     True,   # 换手率（流通市值中性化）
    "amihud":                  True,   # Amihud 非流动性
    "est_peg":                 True,   # 预测 PEG（2026-08 新增）
    ...
}
```

### 因子方向（`FACTOR_DIRECTIONS`）

多因子合成前对负向因子乘以 -1，统一方向（高分 = 好股票）：

```python
FACTOR_DIRECTIONS = {
    "turnover_20": -1,   # 低换手率好
    "amihud":      +1,   # 高 Amihud 好（小盘流动性溢价）
    ...
}
```

### 缓存控制

```python
BACKTEST_CONFIG = {
    "force_recalc": True,   # 当前为 True。以下三种情况必须置 True：
                            #   ① Database 新增交易日  ② 数据源切换
                            #   ③ 改了因子逻辑或回测区间
}
```

---

## 因子列表

### 微观结构因子

| 因子 | 说明 | 中性化 | 方向 |
|------|------|--------|------|
| `reversal_20` | 短期反转（20日） | 无 | 负 |
| `momentum_12_1` | 中期动量（12-1月） | 无 | 负（A股失效） |
| `turnover_20` | 换手率（20日均） | 无 | 负 |
| `turnover_20_neutral` | 换手率（20日均） | 流通市值 | 负 |
| `amihud` | Amihud 非流动性（3月滚动） | 无 | 正 |
| `amihud_neutral` | Amihud 非流动性（3月滚动） | 流通市值 | 正 |
| `amihud_zero_adj` | Amihud 零交易日调整版 | 无 | 正 |
| `amihud_zero_adj_neutral` | Amihud 零交易日调整版 | 流通市值 | 正 |
| `cs_spread` | Corwin-Schultz 高低价价差 | 无 | 负 |
| `roll_spread` | Roll 价差 | 无 | 负 |
| `overnight_ret` | 隔夜收益率（月均） | 无 | 正 |
| `volatility_30` | 短期波动率（30日） | 无 | 负 |
| **`ivol`** | **特质波动率（FF3残差年化）** | **无** | **正** |
| **`ff3_betas`** | **市场 Beta（β_MKT）** | **无** | **正** |

### 基本面因子

| 因子 | 说明 | 中性化 | 方向 |
|------|------|--------|------|
| `pb` | 市净率（原始） | 无 | 负 |
| `bm` | 账面市值比 log(1/PB) | 无 | 正 |
| `pe_ttm` | 市盈率 TTM | 无 | 负 |
| `pe1` | 市盈率（⚠️ 2026-08 起为 Choice PE 静态口径，名字是历史遗留） | 无 | 负 |
| `dividend_yield` | 股息率 TTM | 无 | 正 |
| `ps_ttm` | 市销率 TTM | 无 | 负 |
| `ev_ebitda` | 企业倍数 EV2/EBITDA | 无 | 负 |
| `est_pe_ftm` | 预测市盈率（未来12月）⚠️ 覆盖约 51% | 无 | 负 |
| `est_peg` | 预测 PEG ⚠️ 覆盖约 51% | 无 | **正**（实测方向与直觉相反） |
| `ev2_neutral` | log(企业价值剔除货币资金) | **流通市值** | 负 |
| `size` | log(总市值) | 无 | 负 |
| `size2` | log(流通市值) | 无 | 负 |
| `net_profit_yoy` | 净利润同比增速 | 流通市值+行业 | 正 |

详细构建方法见 [`src/factors/FACTORS.md`](src/factors/FACTORS.md)。

---

## 回测输出说明

| 文件 | 说明 |
|------|------|
| `output/<factor>_group_returns.csv` | 各分组月度收益序列 |
| `output/<factor>_ic_series.csv` | 月度 IC 序列 |
| `output/<factor>_summary.csv` | 分组绩效统计（年化收益、Sharpe、最大回撤等） |
| `output/img/<factor>_nav.png` | 分组累计净值曲线（上：对数坐标；下：LS 多空） |

---

## 数据来源

| 来源 | 路径 | 覆盖 |
|------|------|------|
| Database（主） | `/Users/louis/MyProjects/Database/data/stock/A/` | **2005-01-04 ~ 2026-08-14，5861 只股票，5250 个交易日** |
| 存档（辅） | `_archive/raw_data/` | 2014-2021，仅 `net_profit` 仍依赖 |

**A 股 29 个字段已于 2026-08 全部迁至 Choice（东方财富），Datayes / iFinD 均已退出 A 股链路。**
`.sources.json` 现为 `Counter({'choice': 29})`。维护命令在 Database 项目：

```bash
cd /Users/louis/MyProjects/Database
python3 src/update.py                       # 周更总入口
python3 src/choice_derive.py --promote      # 长表 → 宽表并生效到 data/stock/A/
```

---

## 回测结果（2007-01 ~ 2026-07，五分组，232~234 个月，26 个因子）

> 2026-08-17 全量跑通 26/26。数据源已 100% Choice。
> 重新运行后 `output/` 下 CSV 与 PNG 自动更新。

### 正向因子（IC > 0）

| 因子 | RankIC | ICIR | LS 年化 | LS 夏普 |
|------|--------|------|---------|---------|
| ★ `est_peg` | **0.0570** | 0.461 | **+18.60%** | **1.579** |
| `amihud` | 0.0556 | 0.369 | +18.66% | 1.102 |
| `bm` | 0.0544 | 0.317 | +8.37% | 0.417 |
| `amihud_zero_adj` | 0.0440 | 0.262 | +14.52% | 0.760 |
| `dividend_yield` | 0.0353 | 0.301 | +5.51% | 0.397 |
| `amihud_zero_adj_neutral` | 0.0230 | 0.158 | +5.61% | 0.344 |
| `amihud_neutral` | 0.0229 | 0.195 | +5.26% | 0.457 |
| `overnight_ret` | 0.0160 | 0.215 | +2.77% | 0.327 |

### 负向因子（IC < 0，低值 → 高收益；列 G1 组绩效）

| 因子 | RankIC | ICIR | G1 年化 | G1 夏普 |
|------|--------|------|---------|---------|
| `turnover_20_neutral` | **-0.0980** | **-0.694** | **+23.32%** | **0.776** |
| `turnover_20` | -0.0699 | -0.385 | +14.95% | 0.564 |
| `volatility_30` | -0.0680 | -0.381 | +12.03% | 0.450 |
| `ivol` | -0.0659 | -0.382 | +12.08% | 0.460 |
| `reversal_20` | -0.0640 | -0.438 | +13.84% | 0.409 |
| `size` | -0.0630 | -0.342 | +26.94% | 0.758 |
| `cs_spread` | -0.0577 | -0.441 | +14.02% | 0.472 |
| `pb` | -0.0538 | -0.314 | +15.62% | 0.504 |
| `roll_spread` | -0.0535 | -0.487 | +13.15% | 0.435 |
| `size2` | -0.0479 | -0.257 | +21.86% | 0.603 |
| ★ `est_pe_ftm` | -0.0395 | -0.212 | +12.94% | 0.434 |
| ★ `ev_ebitda` | -0.0375 | -0.223 | +13.89% | 0.460 |
| ★ `ps_ttm` | -0.0352 | -0.241 | +14.37% | 0.464 |
| ★ `ev2_neutral` | -0.0340 | **-0.409** | +17.27% | 0.525 |
| `pe1` | -0.0329 | -0.222 | +13.53% | 0.455 |
| `pe_ttm` | -0.0326 | -0.208 | +12.53% | 0.426 |
| `momentum_12_1` | -0.0158 | -0.102 | +6.20% | 0.197 |
| `ff3_betas` | -0.0084 | -0.071 | +4.42% | 0.152 |

★ = 2026-08-17 新增。`ff3_betas` 近乎无预测力，实际起对照组作用——
它证明框架没有凭空造出信号。

### 多因子合成（10 因子 = 7 微观 + 3 基本面，232 个月）

| 模式 | IC均值 | ICIR | LS年化 | LS Sharpe | LS MDD | 胜率 |
|------|--------|------|--------|-----------|--------|------|
| 等权合成 | 0.1104 | **0.762** | **+24.15%** | **1.349** | -34.5% | 69.8% |
| ICIR权重（12M） | 0.1097 | 0.752 | +24.77% | 1.379 | -30.5% | 71.6% |

合成池：`turnover_20_neutral` / `reversal_20` / `roll_spread` / `cs_spread` /
`amihud` / `volatility_30` / `overnight_ret` + `size` / `bm` / `pe1`。
**2026-08 新增的 5 个因子均未入池**——先看单因子表现，避免把「换源」与「加因子」
两件事的因果混在一起。

---

## 使用这些结果前必读

**`est_peg` 是风格暴露，不是稳定 alpha**

它 RankIC 全场最高，但方向是 **高 PEG 跑赢**（贵的成长股赢），与经济直觉相反。
分期 IC：2007-11 `0.0488` / 2012-16 `0.0473` / **2017-21 `0.0853`** / 2022-26 `0.0453`。
峰值窗口恰好是 A 股「核心资产 / 白马」行情期，特征严丝合缝——
这是 **regime-dependent** 的风格押注。

已排除的：分组塌陷（G1~G5 各 233 期）、薄截面（仅 2007-01/02 不足 100 只，
剔除后 IC 反升至 0.0599）、size 马甲（截面秩相关仅 0.16）。
**无法排除的**：分析师预测数据的全历史重述——严格的点位检验需要历史快照，
本地数据做不到。入合成池前应补分市值域与分市场状态的稳健性检验。

**`pe1` 的口径已变，历史值不可比**

2026-08-17 起 `pe1` 读的是 Choice `PE`（最近年报静态），而非原 Datayes 的
单季年化动态 PE（= 市值 ÷ 最新单季净利 × 4）。Choice 无净利润字段，
重建不出旧口径（新旧秩相关仅 0.20~0.43）。因子名是历史遗留，
**该因子的数字不能与 2026-08-17 之前的回测比较**。

**`ev2_neutral` 的中性化是必需步骤**

原始 `EV2` 是以元为单位的水平量（中位约 62 亿），不是比率。
不做市值中性化的话，它只是 `size` 的复制品。
