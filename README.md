# Quant-CN-markets — A 股量化因子回测框架

模块化因子研究平台，支持因子构建、股票池过滤、分组回测、IC 检验与净值可视化。
数据源为 Database（主）+ `_archive/raw_data`（存档补充）。

---

## 快速开始

```bash
# 激活共享环境（上层目录 venv，Python 3.13）
source /Users/louis/MyProjects/venv/bin/activate

# 运行回测（在 src/main.py 中将需要的因子 FACTOR_FLAGS 设为 True）
# 注意：以包方式运行，须在 Quant/ 根目录执行
cd /Users/louis/MyProjects/Quant
python -m src.main
```

代码以 `src/` 包形式组织，内部统一使用 `from src.xxx import ...` 绝对导入，
因此必须用 `python -m src.main` 运行（不能 `python src/main.py`）。
回测结果保存至 `output/`，净值曲线图保存至 `output/A/img/`、`output/HK/img/`。
依赖见 `requirements.txt`。

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
│   │   └── FACTORS.md         # 因子构建方法文档
│   ├── backtest/
│   │   ├── engine.py          # 月度分组回测核心
│   │   ├── metrics.py         # 绩效指标（IC、ICIR、Sharpe、最大回撤等）
│   │   └── report.py          # 报告输出（控制台 + CSV + PNG）
│   └── strategy/
│       ├── optimizer.py       # 因子打分与分组工具
│       ├── scoring.py         # 因子打分
│       └── combine_factors.py # 多因子合成（方向对齐 → 正交化 → ICIR 权重）
│
├── output/                    # 回测输出 CSV + PNG（不跟踪）
├── reports/                   # HTML / pptx 报告（不跟踪）
└── _archive/                  # 历史代码 + raw_data + 旧脚本 defense/run_backtest（不跟踪）
```

---

## 主要配置（`main.py`）

### 回测区间

```python
BACKTEST_START = "2007-01-01"   # 起始日期（2007起：PB/PE 数据可信度较好）
BACKTEST_END   = <自动取上月末>  # 截止日期（动态，每次运行自动更新）
```

### 因子开关（`FACTOR_FLAGS`）

将需要运行的因子设为 `True`，其余保持 `False`：

```python
FACTOR_FLAGS = {
    "turnover_20":             True,   # 换手率（无中性化）
    "turnover_20_neutral":     False,  # 换手率（流通市值中性化）
    "amihud":                  True,   # Amihud 非流动性
    "amihud_neutral":          False,  # Amihud（流通市值中性化）
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
    "force_recalc": False,  # True = 忽略缓存，强制重算（数据更新后使用）
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

### 基本面因子

| 因子 | 说明 | 中性化 | 方向 |
|------|------|--------|------|
| `pb` | 市净率（原始） | 无 | 负 |
| `bm` | 账面市值比 log(1/PB) | 无 | 正 |
| `pe_ttm` | 市盈率 TTM | 无 | 负 |
| `pe1` | 动态市盈率 | 无 | 负 |
| `dividend_yield` | 股息率 TTM | 无 | 正 |
| `size` | log(总市值) | 无 | 负 |
| `size2` | log(流通市值) | 无 | 负 |
| `net_profit_yoy` | 净利润同比增速 | 流通市值+行业 | 正 |

详细构建方法见 [`factors/FACTORS.md`](factors/FACTORS.md)。

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
| Database（主） | `/Users/louis/MyProjects/Database/data/stock/A/` | 2004-至今，6077只股票 |
| 存档（辅） | `_archive/raw_data/` | 2014-2021，仅 `net_profit` 仍依赖 |

Database 数据由 `Database/codes/ifind.py` 和 `Database/codes/datayes.py` 维护更新。

---

## 已有回测结果（2007-01 ~ 2026-03，五分组，231个月）

> 注：以下为最新运行结果（v2，10因子合成）。重新运行后 output/ 目录下 CSV 自动更新。

**单因子（按 |ICIR| 降序，合成候选标注 ✅）**

| 因子 | IC均值 | ICIR | LS Sharpe | 合成 |
|------|--------|------|-----------|------|
| turnover_20 | -0.097 | -0.715 | 1.36 | ✅ |
| roll_spread | -0.053 | -0.490 | 0.71 | ✅ |
| reversal_20 | -0.064 | -0.450 | 0.99 | ✅ |
| cs_spread | -0.056 | -0.447 | 0.60 | ✅ |
| amihud | +0.059 | +0.387 | 1.20 | ✅ |
| volatility_30 | -0.066 | -0.383 | 0.44 | ✅ |
| size（总市值） | -0.066 | -0.355 | 1.16 | ✅ 基本面 |
| bm（账面市值比） | +0.055 | +0.334 | 0.49 | ✅ 基本面 |
| pe1（动态市盈率） | -0.046 | -0.298 | 0.55 | ✅ 基本面 |
| overnight_ret | +0.017 | +0.222 | 0.33 | ✅ |
| momentum_12_1 | -0.020 | -0.128 | 0.31 | ❌ A股失效 |

**多因子合成（v2：10因子 = 7微观 + 3基本面，等权 vs ICIR权重）**

| 模式 | IC均值 | ICIR | LS年化 | LS Sharpe | LS MDD | 胜率 |
|------|--------|------|--------|-----------|--------|------|
| 等权合成 | 0.114 | **0.815** | **+26.9%** | **1.79** | -16.8% | 72.6% |
| ICIR权重（12M） | 0.112 | 0.798 | +26.3% | 1.69 | -18.2% | 71.3% |

*v1（旧，9微观因子，2014~2024）：ICIR 0.77 / LS年化 +21.4% / LS Sharpe 1.58*
