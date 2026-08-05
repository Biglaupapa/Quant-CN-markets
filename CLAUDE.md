# Quant-CN-markets — A 股量化因子回测框架

A 股量化因子研究平台，支持因子构建、股票池过滤、分组回测、IC 检验和净值可视化。

---

## 快速开始

### 运行回测

```bash
source /Users/louis/MyProjects/venv/bin/activate   # 共享 venv，Python 3.13
cd /Users/louis/MyProjects/Quant
python -m src.main                                  # 以包方式运行（须在 Quant/ 根目录）
```

代码以 `src/` 包形式组织，内部统一 `from src.xxx import ...`，必须用 `python -m src.main` 运行（不能 `python src/main.py`）。依赖见 `requirements.txt`。

控制哪些因子参与回测：修改 `src/main.py` 顶部的 `FACTOR_FLAGS` 字典（`True` = 计算，`False` = 跳过）。

### 构建 FF3 因子

当 Database 数据更新后，手动运行 FF3 构建程序：

```bash
python -m src.ff3_builder                          # 全量重建 FF3 因子
python -m src.ff3_builder --dry-run                # 查看当前 FF3 文件状态
```

**工作流**：
1. 修改 Database 中的量价数据或宏观数据（bond_yield_1y）
2. 运行 `python -m src.ff3_builder` 构建 FF3
3. FF3 输出到 `/Users/louis/MyProjects/Database/data/factors/`
4. 运行 `python -m src.main` 执行回测（自动读取最新 FF3）

---

## 项目结构

```
Quant/                       # git 跟踪：仅 src/ + 项目文件（数据/结果不跟踪）
├── requirements.txt         # 依赖说明（共用上层 venv）
├── README.md  CLAUDE.md  .gitignore
├── src/                     # ★ 源码包
│   ├── __init__.py
│   ├── main.py              # ★ 回测入口（python -m src.main）
│   ├── ff3_builder.py       # ★ FF3 因子构建（python -m src.ff3_builder）
│   ├── convert_report_to_pdf.py
│   ├── config/settings.py   # 全局路径、字段映射、因子参数
│   ├── data/
│   │   ├── loader.py        # 统一数据加载（Database 优先，存档补充）
│   │   └── universe.py      # 股票池过滤（ST / 停牌 / 次新股）
│   ├── factors/
│   │   ├── base.py          # 预处理流水线
│   │   ├── microstructure.py# 微观结构因子
│   │   └── fundamental.py   # 基本面因子
│   ├── backtest/
│   │   ├── engine.py        # 月度分组回测核心
│   │   ├── metrics.py       # 绩效指标（IC、ICIR、Sharpe、最大回撤）
│   │   └── report.py        # 报告输出（控制台 + CSV + 净值曲线 PNG）
│   └── strategy/
│       ├── optimizer.py     # 因子打分与分组
│       ├── scoring.py       # 因子打分
│       └── combine_factors.py # 多因子合成
│
├── output/                  # 回测结果（不跟踪）
│   ├── cache/               # 因子缓存，命中则跳过重新计算
│   ├── A/stats/  A/img/     # A 股 CSV 统计 + 净值曲线 PNG
│   └── HK/stats/ HK/img/    # 港股
├── reports/                 # HTML / pptx 报告（不跟踪）
└── _archive/                # 历史代码 + raw_data + 旧脚本（不跟踪）
                             # raw_data 仅 net_profit 因子仍依赖
```

---

## 数据源

### 主数据源：Database
路径：`/Users/louis/MyProjects/Database/data/stock/A/`

> **⚠️ 2026-08 数据源变更**：A 股 19 个字段已由 iFinD/Datayes 切换到 **Choice**。
> 起点由 2003-01-02 改为 **2005-01-04**（`settings.py` 的 `DB_START` 需同步改），
> 股票列由 6077 改为 **5855**（时点并集，无幸存者偏差）。
> 详见 `Database/docs/【主文档】框架全景记录与Choice迁移方案.md`。

**Choice 供给（19 个，覆盖 2005-01-04 ~ 2026-08-05 / 5855 只 / 5231 个交易日）**

| 字段（loader 名） | 文件 | 用途 |
|---|---|---|
| `close_adj` / `open_adj` | `close_adj.csv` / `open_adj.csv` | 月度收益率 |
| `high_adj` / `low_adj` | `high_adj.csv` / `low_adj.csv` | 复权高低价因子 |
| `close` | `close.csv` | 市值计算（= `MV/TOTALSHARE`，精确恒等） |
| `open` / `high` / `low` | 对应 csv | 不复权价（`USE_ADJ_PRICE=False` 时） |
| `turn` | `turn.csv` | 换手率因子（= `VOLUME/LIQSHARE×100`，全精度） |
| `amt` / `volume` | `amt.csv` / `volume.csv` | Amihud 因子 |
| `is_st` | `st.csv` | 股票池过滤（`ISSTSTOCK` **或** `ISXSTSTOCK`） |
| `trade_status` | `status.csv` | 股票池过滤（1=有成交，0=停牌） |
| `float_shares` | `free_float_shares.csv` | 市值中性化 |
| `market_value` / `neg_market_value` | 对应 csv | Size / Size2 因子 |
| — | `xst.csv` | **新增**：单独的 \*ST 标记，loader 尚未映射 |
| — | `vwap.csv` / `vwap_adj.csv` | 本框架未使用 |

**Datayes 供给（4 个，覆盖至 2026-07-17）**

| 字段 | 文件 | 用途 | 备注 |
|---|---|---|---|
| `pb` | `pb.csv` | PB 因子 | Choice 的 `PB` 口径与此相差 8~29%，未切换 |
| `pe_ttm` | `pe_ttm.csv` | PE 因子 | Choice 的 `PETTM` 已验证逐位相等，可随时切 |
| `pe1` | `pe1.csv` | 动态 PE | **Choice 无对应字段** |
| `dividend_ratio` | `dividend_ratio.csv` | 股息率因子 | Choice 的 `LASTESTDIVIDEND` 更完整，可随时切 |

**iFinD 供给（1 个）**

| 字段 | 文件 | 用途 | ⚠️ 已知问题 |
|---|---|---|---|
| `listing_days` | `listed_days.csv` | 股票池过滤（次新股） | 列 5828（**比 Choice 少 48 只**）、止于 **2026-07-17**（落后 8 天）。`build_investable_mask` 取三者列交集 = 5803，且末端日期会被静默剔除 |

**宏观与因子**

| 字段 | 文件 | 覆盖 | 用途 |
|---|---|---|---|
| `bond_yield_1y` | `macro/bond_yield_1y.csv` | 2004-01-02 ~ 2026-07-17 | 日度国债收益率（% 形式）→ `ff3_builder` 现算 Rf |
| FF3 月度 | `factors/ff3_monthly.csv` | 待用新数据重建 | 市场/规模/价值因子 |
| FF3 日度 | `factors/ff3_daily.csv` | 待用新数据重建 | 日度 FF3 + `Rf` 列 |

> `rf_daily.csv` **已于 2026-08-05 删除**。它是 `bond_yield_1y/100/252` 的派生（5630 点逐点全等），
> 带表头 bug，且全库无任何代码读取——IVOL / ff3_betas 实际读的是 `ff3_daily.csv` 的 `Rf` 列。

### 补充来源：存档（_archive/raw_data/）
仅用于 `net_profit`（归母净利润，季度）字段，供 `net_profit_yoy` 因子使用。覆盖 2014~2021。

---

## 回测参数配置（main.py）

```python
BACKTEST_START = "2014-01-01"   # 回测起始日期
BACKTEST_END   = "2024-12-31"   # 回测截止日期（None = 运行当天）

BACKTEST_CONFIG = {
    "n_groups":     5,      # 分组数（5 或 10）
    "freq":         12,     # 数据频率（月度=12，季度=4）
    "save_output":  True,   # 保存 CSV 和 PNG 至 output/
    "force_recalc": False,  # True = 忽略缓存强制重算（数据更新后使用）
}
```

---

## 因子清单与状态

### 微观结构因子（factors/microstructure.py）

| 因子 | 说明 | 状态 | 所需数据 |
|---|---|---|---|
| `turnover_20` | 换手率（20日均，市值中性化） | ✅ 已验证 | turn, close, float_shares |
| `reversal_20` | 短期反转（20日累计收益） | 待验证 | close_adj |
| `momentum_12_1` | 中期动量（12-1月） | 待验证 | close_adj |
| `amihud` | Amihud 非流动性（3月滚动） | 待验证 | close_adj, amt |
| `amihud_zero_adj` | Amihud 零交易日调整版 | 待验证 | close_adj, amt |
| `cs_spread` | Corwin-Schultz 高低价价差 | 待验证 | high, low |
| `roll_spread` | Roll 价差 | 待验证 | close_adj |
| `overnight_ret` | 隔夜收益率（月均） | 待验证 | close_adj, open_adj |
| `volatility_30` | 短期波动率（30日） | 待验证 | close_adj |

### 基本面因子（factors/fundamental.py）

| 因子 | 说明 | 状态 | 所需数据 |
|---|---|---|---|
| `pb` | 市净率（市值+行业双重中性化） | 待验证 | pb（待补全） |
| `pe_ttm` | 市盈率 TTM | 待验证 | pe_ttm（2023+） |
| `dividend_yield` | 股息率 TTM | 待验证 | dividend_ratio（待补全） |
| `size` | 市值因子 log(流通市值) | 待验证 | close, float_shares |
| `net_profit_yoy` | 净利润同比增速 | 待验证 | net_profit（存档，2014~2021） |

### 待激活因子（需额外数据）

| 因子 | 状态 | 所需数据 |
|---|---|---|
| `ps_gamma` | ❌ | marketrtn_daily.csv（日度市场收益率） |
| `ps_liq_beta` | ❌ | ps_gamma 先激活 + marketrtn_daily.csv |
| `ap_betas` | ❌ | marketrtn_daily.csv |
| `capm_beta` | ❌ | marketrtn_daily.csv |
| **`ivol`** | **✅ 就绪** | **FF3 日度 + rf_daily（2026-07-15 已补全）** |
| **`ff3_betas`** | **✅ 就绪** | **FF3 日度 + rf_daily（2026-07-15 已补全）** |

---

## 因子激活状态（2026-07-15 更新）

### ✅ 已激活因子

**新激活的两个 Beta 类因子**：
- `ivol`（特质波动率）✅ - 12月滚动 FF3 回归残差标准差
- `ff3_betas`（三因子 Beta）✅ - 市场/规模/价值因子 Beta（β_MKT 为主）

**激活时机与数据就绪**：
- 时间：2026-07-15
- 先决条件：FF3 日度因子（ff3_daily.csv）+ 无风险利率（rf_daily.csv）
- 覆盖范围：2004-01-02 ~ 2026-07-10（日度），月度月末取值

## 已验证因子结果

### turnover_20（2014~2021，存档数据）
IC均值=-0.097，ICIR=-0.72，|IC|>0.02占比=94%
G1年化+22.5%（Sharpe=0.76）vs G5年化-2.1%（Sharpe=-0.06）
**负向因子**：低换手率 → 高未来收益

> 注：升级至全 Database 数据源后（2014~2024），结果待重新验证。

### IVOL 与 FF3 Betas（待回测验证）
两个因子已于 2026-07-15 实现并激活，计算逻辑验证通过。
具体回测表现需运行 `python -m src.main` 后观察。

---

## 开发注意事项

**前瞻偏差（Look-ahead Bias）**
因子在 T 月末排序，预测 T+1 月收益。代码中已正确实现：
```python
fwd_ret = monthly_ret.shift(-1)   # T+1月收益
grp_ret = group_return(factor, fwd_ret, n_groups=n_groups)
ic_series = calc_ic(factor, fwd_ret, method="spearman")
```
不要把 `monthly_ret`（当期收益）直接传入 `group_return`，否则产生前瞻偏差。

**trade_status 格式兼容**
存档：字符串 `"交易"` = 正常；Database：数值 `1` = 正常，`0` = 停牌。
`universe.py` 已兼容两种格式，不需要在因子代码中额外处理。

**因子缓存失效场景**
以下情况需将 `force_recalc` 设为 `True`：
- Database 数据更新（新增交易日）
- 修改了因子计算逻辑
- 修改了回测时间范围

**net_profit 字段**
唯一仍依赖存档的字段。`net_profit_yoy` 因子回测区间受限于存档（2014~2021）。
