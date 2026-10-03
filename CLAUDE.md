# Quant-CN-markets — A 股量化因子回测框架

A 股量化因子研究平台，支持因子构建、股票池过滤、分组回测、IC 检验和净值可视化。

---

## 快速开始

### 运行回测

```bash
conda activate /Users/louis/MyProjects/venv   # 共享环境（conda，虽然目录名叫 venv），Python 3.13
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

**Choice 供给（全部 29 个，覆盖 2005-01-04 ~ 2026-08-14 / 5861 只 / 5250 个交易日）**

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
| `dividend_ratio` | `dividend_ratio.csv` | 股息率因子（= `LASTESTDIVIDEND`，2026-08-17 切换） |
| `pb` | `pb.csv` | PB / BM 因子（= `PB`，2026-08-17 切换） |
| `pe_ttm` | `pe_ttm.csv` | PE 因子（= `PETTM`，2026-08-17 切换） |
| `pe1` | **`pe.csv`** | PE 因子（= `PE`，2026-08-17 切换）⚠️ 键名是历史遗留，见下 |
| `ps_ttm` | `ps_ttm.csv` | PS 因子（= `PSTTM`）覆盖 100% |
| `ev2` | `ev2.csv` | `ev2_neutral` 因子（= `EV2`，单位元，须中性化） |
| `ev_ebitda` | `ev_ebitda.csv` | 企业倍数因子（= `EVTOEBITDA`）覆盖 98% |
| `est_pe_ftm` | `est_pe_ftm.csv` | 预测 PE 因子（= `ESTPEFTM`）⚠️ 覆盖仅 51% |
| `est_peg` | `est_peg.csv` | 预测 PEG 因子（= `ESTPEG`）⚠️ 覆盖仅 51% |
| — | `xst.csv` | **新增**：单独的 \*ST 标记，loader 尚未映射 |
| — | `vwap.csv` / `vwap_adj.csv` | 本框架未使用 |

**Datayes 已于 2026-08-17 完全退场** —— `.sources.json` 现为 `Counter({'choice': 29})`，
无任何非 Choice 项。原 `pe1.csv` 已归档至 `data/stock/A/_retired_datayes/`。

> ⚠️ **`pe1` 这个因子名是历史遗留，口径已变。**
> 原 Datayes `pe1` = 市值 ÷（最新单季净利 × 4），是**单季年化动态 PE**。反推验证
> （2024-06-28）：五粮液 2024Q1 净利 140.5 亿 ×4 = 562 亿 ÷ 市值约 4970 亿 → 8.85，
> 与文件值吻合；茅台同法得 19.15，同样吻合。
>
> Choice 的 26 个字段里**没有净利润**，重建不出该口径。与各列秩相关：
>
> | 对比列 | 2010 | 2016 | 2020 | 2026 |
> |---|---|---|---|---|
> | `PE` | 0.43 | 0.35 | 0.20 | 0.42 |
> | `PETTM` | 0.50 | 0.43 | 0.30 | 0.47 |
> | `ESTPEFTM` | 0.64 | 0.34 | 0.37 | 0.46 |
>
> 决定改用 Choice `PE`（静态 PE）顶替并沿用因子名。**这是语义替换而非无损换源**，
> 该因子历史值已整体改变，不能与 2026-08-17 之前的回测数字比较。
> Database 侧文件如实命名为 `pe.csv`，映射写在 `DATABASE_FIELD_MAP`。

**`listing_days` 已于 2026-08-11 切换到 Choice**

由 `Database/data/listed_days_src/`（Choice `IPOLSTDAYS`，上市**自然日**）反推上市日期，
再套交易日历换算成**上市交易日**，形状 5243 × 5855，与其余 19 个宽表完全对齐。

原 iFinD 版存在两个缺陷，已一并解决：
- 2026-05-25 ~ 07-01 共 **27 个交易日塌陷**，每天仅 2 只股票有值（正常 5828），
  期间股票池会几乎全空且不报错
- 退市股冻结最后值，能通过 `>= 60` 过滤；新版退市后为 NaN

`build_investable_mask` 三输入现同形状：列交集 5803 → **5855**，末端 07-17 → **08-05**。

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
BACKTEST_START = "2007-01-01"   # 回测起始日期
#   数据本身从 2005-01-04 起可用；选 2007 是主动判断——2005~2006 A 股仅
#   1300~1400 只、股改期间大量停牌、估值数据质量差。要改就是这一行。

BACKTEST_END = str((pd.Timestamp.today().to_period("M") - 1).to_timestamp("M").date())
#   自动取上一个完整月末，每次运行自动更新。例：2026-08-11 跑 → 2026-07-31

BACKTEST_CONFIG = {
    "n_groups":     5,      # 分组数（5 或 10）
    "freq":         12,     # 数据频率（月度=12，季度=4）
    "save_output":  True,   # 保存 CSV 和 PNG 至 output/
    "force_recalc": True,   # 忽略缓存强制重算（数据源迁移后置 True，见文末）
}
```

### ★ 回测口径与组建日股票池（2026-10-01 起与文献对齐）

**权威文档：`docs/【方法】回测口径与文献对齐.md` §十（逐条：方法 / 原因 / 依据 / 执行差异）。**

**组建（t 月末，只用 t 月及以前信息）** —— `universe.build_formation_mask()`，参数 `settings.FORMATION_CONFIG`：

| # | 规则 | 依据 |
|---|---|---|
| a | 样本：代码 60 / 00 / 30（不含科创板、北交所） | LSY 2019 |
| b | t 月最后交易日有成交 | HQZ |
| c | t 月成交 ≥ 15 天；过去 12 个月 ≥ 120 天 | LSY |
| d | 上市满 6 个月（日历，替换原 `listed_days ≥ 60`） | LSY、HQZ |
| e | t 月最后交易日非 ST / *ST | 顾明等 |
| f | t 月末不处于待退市（退市整理期，Choice `sector 001026`） | 顾明等 |
| g | 剔除 A 股市值最小 30%（收盘 × A 股总股本含非流通股，财汇） | LSY、HQZ |

**持有（t+1 月）** —— `engine.calc_monthly_returns()`：收益 = t+1 月末 / t 月末后复权收盘价 − 1（收盘到收盘）；
持有期**不再筛选**（ST、停牌照算，停牌按最后价）；退市算到最后交易日，不打折。

**逐日清洗掩码** `build_investable_mask(freq="D")`（status / ST / `listed_days ≥ 60`）**只用于因子计算输入**，不决定入组。

**影响（合成等权，2007-01 ~ 2026-09）**：旧口径夏普 1.32 → 第 1 步（收益侧）1.45 → 第 2 步（股票池）**0.98**。
几乎全部下降来自规则 g：被剔除的最小 30% 等权年化 25.0%（其余 8.9%），占合成多头组 46%、`size` 多头组 99.9%、
`amihud` 多头组 68.7%。剔除后 `size` 夏普 1.07 → 0.28、`amihud` 1.03 → 0.39，`pe1` 0.29 → 0.55（变强）。

> ⚠️ **下面「因子清单与状态」「多因子合成」两节的数字是旧口径**（2026-08，开盘入场 + 持有期筛选 + 旧股票池），
> 保留作历史对照。新口径的合成池 10 因子与合成结果见方法文档 §九 / §十；**全部 62 个因子的新口径重跑尚未做**（待办 #18）。

---

## 因子清单与状态（2026-08-17 全量跑通，26/26）——⚠️ 旧口径，见上节

以下为 `FACTOR_FLAGS = True` 的 26 个因子，按 **RankIC 均值**排序（A 股，2007-01 ~ 2026-07，5 分组）。
★ = 2026-08-17 新增。

### 正向因子（IC > 0）

| 因子 | RankIC | ICIR | LS 年化 | LS 夏普 | 所需数据 |
|---|---|---|---|---|---|
| ★ `est_peg` | **0.0570** | 0.461 | **+18.60%** | **1.579** | est_peg ⚠️ 覆盖 51% |
| `amihud` | 0.0556 | 0.369 | +18.66% | 1.101 | close_adj, amt |
| `bm` | 0.0544 | 0.317 | +8.37% | 0.417 | pb |
| `amihud_zero_adj` | 0.0440 | 0.262 | +14.52% | 0.760 | close_adj, volume |
| `dividend_yield` | 0.0353 | 0.301 | +5.51% | 0.397 | dividend_ratio |
| `amihud_zero_adj_neutral` | 0.0230 | 0.158 | +5.60% | 0.343 | + neg_market_value |
| `amihud_neutral` | 0.0229 | 0.195 | +5.25% | 0.457 | + neg_market_value |
| `overnight_ret` | 0.0160 | 0.215 | +2.77% | 0.327 | close_adj, open_adj |

### 负向因子（IC < 0，低值 → 高收益）

| 因子 | RankIC | ICIR | G1 年化 | G1 夏普 | 所需数据 |
|---|---|---|---|---|---|
| `turnover_20_neutral` | **-0.0980** | **-0.694** | **+23.32%** | **0.776** | turn, neg_market_value |
| `turnover_20` | -0.0699 | -0.385 | +14.95% | 0.564 | turn |
| `volatility_30` | -0.0681 | -0.381 | +12.04% | 0.450 | close_adj |
| `ivol` | -0.0659 | -0.382 | +12.08% | 0.461 | close_adj + ff3_daily |
| `reversal_20` | -0.0640 | -0.438 | +13.84% | 0.410 | close_adj |
| `size` | -0.0630 | -0.342 | +26.94% | 0.758 | market_value |
| `cs_spread` | -0.0578 | -0.441 | +14.02% | 0.472 | high_adj, low_adj |
| `pb` | -0.0538 | -0.314 | +15.62% | 0.504 | pb |
| `roll_spread` | -0.0535 | -0.487 | +13.15% | 0.435 | close_adj |
| `size2` | -0.0479 | -0.257 | +21.86% | 0.603 | neg_market_value |
| ★ `est_pe_ftm` | -0.0395 | -0.212 | +12.94% | 0.434 | est_pe_ftm ⚠️ 覆盖 51% |
| ★ `ev_ebitda` | -0.0375 | -0.223 | +13.89% | 0.460 | ev_ebitda |
| ★ `ps_ttm` | -0.0352 | -0.241 | +14.37% | 0.464 | ps_ttm |
| ★ `ev2_neutral` | -0.0340 | **-0.409** | +17.27% | 0.525 | ev2 + neg_market_value |
| `pe1` | -0.0329 | -0.222 | +13.53% | 0.455 | pe1→`pe.csv` ⚠️ 口径已变 |
| `pe_ttm` | -0.0326 | -0.208 | +12.53% | 0.426 | pe_ttm |
| `momentum_12_1` | -0.0158 | -0.102 | +6.20% | 0.197 | close_adj |
| `ff3_betas` | -0.0084 | -0.072 | +4.42% | 0.152 | close_adj + ff3_daily |

> `ff3_betas` 近乎无预测力（LS -0.23%），符合预期——β 在 A 股本就不是有效因子。
> 它实际起**对照组**作用：框架没有凭空造出信号。

### ✅ `dividend_yield` 分组塌陷已修复（2026-08-17）

**症状**：各组非空期数 `G1=234 G2=234 G3=234 G4=178 G5=5`——G5 只有 5 个月，
LS 夏普虚高到 1.96、最大回撤仅 -1.45%，绩效数字全部不可用。

**根因**：Datayes 的 `divRateL12m` 把**缺失填成 0**（不是真的不分红）。
截面零值率 22%~53%，25 分位就是 0.0000，标准化后大量股票挤在同一个值，
`qcut` 分位边界重复 → 高分位组被挤空。

**修法**：切到 Choice 的 `LASTESTDIVIDEND`。该列一直躺在
`Database/data/stock/A2005-2026/` 的原始月度文件里，只是 P4 迁移时漏了抽取。
在 `Database/src/choice_derive.py` 的 SPECS 里补一条 L0 纯透视即可（同步登记进主文档 §3.7）：

```bash
python3 src/choice_derive.py --only dividend_ratio --out-dir <暂存> --promote
```

**验证**：

| 截面日期 | 旧零值率 | 新零值率 |
|---|---|---|
| 2008-06-30 | 53.4% | 0.0% |
| 2012-12-31 | 32.0% | 0.0% |
| 2018-08-31 | 21.9% | 0.0% |
| 2024-01-31 | 33.9% | 0.0% |
| 2026-07-17 | 33.3% | 0.0% |

修复后各组均为 234 期，G1→G5 年化单调递增（6.73% → 15.27%），
IC 0.0213 → **0.0353**，ICIR 0.197 → **0.301**，LS 夏普回落到真实的 0.397。
其余 20 个因子与合成结果逐位未变（该因子不在合成池）。

### ⚠️ `pe_ttm` / `pb` 换 Choice 后的历史值变化（2026-08-17）

两者都已切到 Choice（`PETTM` / `PB`，L0 纯透视，负值原样保留——「亏损公司置 NaN」
在 `fundamental.py` 里做，不在 Database 层）。**但换源不是无损的**，实测新旧对拍：

| 字段 | 截面 | 相对差 <1e-6 占比 | 中位相对差 | 秩相关 |
|---|---|---|---|---|
| `pe_ttm` | 2026-07-17 | 66.2% | 0.00% | **0.9980** |
| `pe_ttm` | 2020-06-30 | 60.9% | 0.00% | **0.9746** |
| `pe_ttm` | **2012-12-31** | **0.0%** | **18.58%** | **0.5235** |
| `pb` | 2026-07-17 | 0.0% | 1.80% | 0.9849 |
| `pb` | 2020-06-30 | 0.0% | 1.95% | 0.9806 |
| `pb` | 2012-12-31 | 6.6% | 0.00% | 0.9977 |

**关键**：主文档 §3.3 记的「`PETTM` 已验证逐位相等」**只在近年成立**。2012 年截面
秩相关仅 0.52、中位相对差 18.6%——两家对早期财报的口径差异很大。因此
`pe_ttm` 因子的**早期历史结果已经变了**，不能与迁移前的数字直接比较。

`pb` 则全区间秩相关稳定在 0.98~0.998，变化温和。

因子层面的实际影响（`pe_ttm` 覆盖期数 220 → **234**，不再被 Datayes 的短历史截断）：

| 因子 | 迁移前 | 迁移后 |
|---|---|---|
| `pe_ttm` IC / ICIR | -0.0376 / -0.237 | **-0.0326 / -0.208** |
| `pb` IC / ICIR | -0.0548 / -0.324 | -0.0538 / -0.314 |
| `bm` IC / ICIR | 0.0551 / 0.325 | 0.0544 / 0.317 |
| 合成等权 LS 年化 / 夏普 | +24.84% / 1.383 | +24.60% / 1.374 |

### ★ 2026-08-17 新增 5 个 Choice 估值因子

全部跑通、**无一分组塌陷**（G1~G5 各组非空期数极差均为 0）。

**`est_peg` 是本轮最大发现，但方向与直觉相反**

RankIC **+0.0570，全部 26 个因子里最高**；LS 年化 +18.60%、夏普 **1.579**（仅次于合成因子）。
但方向是 **+1 而非 -1**——高 PEG 组（G5）年化 +16.94%，低 PEG 组（G1）反而 -3.23%。
`FACTOR_DIRECTIONS` 已按实测回填为 `+1`。

已排除三类假象：

| 检验 | 结果 |
|---|---|
| 薄截面导致的噪声 | 仅 2007-01/02 两个月不足 100 只；**剔除后 IC 反升**至 0.0599、ICIR 0.508 |
| 分组塌陷 | G1~G5 各 233 期，完全均衡 |
| 只是 size 的马甲 | 与 `size` 截面秩相关均值仅 **0.16**；与 `est_pe_ftm` 仅 0.31 |

⚠️ 但仍须保留一条：覆盖率只有 51%，**只有分析师跟踪的标的才有值**，而这半个市场
偏大盘、非随机抽样。该因子的多空组合实际只在被覆盖的股票里交易，流动性天然更好，
这可能部分解释了偏高的夏普。入合成池前建议先做分市值域的稳健性检验。

**`ev2_neutral` 的 ICIR 是新因子里第二好的（-0.409）**

原始 `EV2` 是以元为单位的水平量（中位约 62 亿），直接用等于市值代理。
取 `log` 后对 `log(流通市值)` 回归取残差，剥离规模成分。中性化是必需步骤而非可选项。

**`ps_ttm` / `ev_ebitda` / `est_pe_ftm` 表现相近**（RankIC -0.035 ~ -0.040），
与 `pe_ttm` / `pe1` 同属估值大类，信号强度也相当。`ps_ttm` 覆盖 100%、零负值，
是这批里数据质量最好的。

**本轮新因子均未进入合成池**（`COMBINE_FACTORS` 未动），以免把「换源」与「加因子」
两件事的因果混在一起。合成结果的变化只来自 `pe1` 换源。

### 未激活因子

| 因子 | 状态 | 缺什么 |
|---|---|---|
| `net_profit_yoy` | ❌ 关闭 | net_profit 仅存档 2014~2021（Choice 无净利润字段） |

> ✅ **2026-08-18：`ps_gamma` / `ps_liq_beta` / `ap_betas` / `capm_beta` 已全部激活。**
> 原标注的「缺 `marketrtn_daily.csv`」是伪缺口——`ff3_daily.csv` 的 `MKT` 是
> **超额**市场收益，`r_m = MKT + Rf` 即所需序列；`ap_betas` 要的日度流通市值
> 就是 `neg_market_value.csv`。另拆出 `beta_smb` / `beta_hml`（原被 `attrs`
> 吞掉）。因子数 **26 → 37**，未下载任何新数据。详见 `src/factors/FACTORS.md`。

### 2026-08-18 新增 11 个风险因子（摘要）

| 因子 | RankIC | ICIR | LS 夏普 |
|---|---|---|---|
| `ap_beta2` / `ap_beta5` | +0.027 | +0.23 / +0.24 | **0.97 / 1.01** ⭐ |
| `ap_beta1` / `ap_beta3` / `ap_beta4` | -0.012 ~ -0.021 | -0.10 ~ -0.17 | 弱 |
| `capm_beta` / `beta_smb` / `beta_hml` | \|IC\| < 0.018 | \|ICIR\| < 0.11 | ✗ 无预测力 |
| `ps_gamma` / `ps_liq_beta` | \|IC\| < 0.006 | \|ICIR\| < 0.09 | ✗ 无预测力 |

**结论**：β 载荷类在 A 股整体无效（含既有 `ff3_betas`），印证「A 股由**特征**定价
而非**载荷**定价」。唯一有信号的是 AP 的流动性共动 β2/β5。全部 11 个暂不入合成池，
但**应作为特征进入 ML 面板**——单因子线性 IC 低 ≠ 在树模型中无信息量。

**既有结果零影响**：26 个既有因子 RankIC/ICIR 逐项复现，合成等权 LS 年化 24.15%、
夏普 1.3494、胜率 69.83%，与迁移前逐位一致。

### 2026-08-18 特征扩充：新增 25 个因子（`src/factors/expansion.py`）

为 ML 面板准备特征宽度。入选标准：GKX/JKP 有清晰构建方法 + Choice 现有字段可构建
+ **不做估值反推**（财报数据优先，没有就不做）+ 不做行业中性化第二套。
故本批只含纯量价/成交衍生特征。

**三个主要发现**：

| 发现 | 证据 |
|---|---|
| ⭐ `amihud_vwap` **强于既有 `amihud`** | ICIR 0.515 vs 0.369、LS 夏普 1.324 vs 1.100 |
| ⭐ MAX effect 在 A 股很强 | `rmax1_21` ICIR -0.575、`rmax5_21` -0.532 |
| ⭐ 动量**跨全部期限**系统性失效 | 6 个新窗口 \|ICIR\| 全部 < 0.23，多数为负号 |

`amihud_vwap` / `close_vwap_dev` 用的 VWAP 是 **Choice 相对 CRSP/JKP 的独有字段**
（美国学术库不提供日内 VWAP），这类特征在英文文献里几乎见不到。

其余较强的新因子：`dolvol_126`（IC -0.0804，全库第 2）、`high_low_range`、
`rvol_21`、`turn_std_21`（ICIR -0.588）、`skew_21`。
明确无效：`age` / `seasonality` / `ret_12_7` / `float_shares_chg` / `free_float_ratio`。

⚠️ `zero_trades_21` 分组回测不可用（A 股 96%~99.5% 的股票从不停牌 → qcut 分位
边界重复 → division by zero），`FACTOR_FLAGS` 已置 False，**但缓存仍生成，
ML 阶段应重新纳入**。这是数据真实性质，不是 bug。

### 因子总数

| 阶段 | 因子数 |
|---|---|
| 2026-08-17 | 26 |
| + 风险/流动性风险激活（11） | 37 |
| + 特征扩充（25） | **62** |

新增的 36 个因子**一个都没进 `COMBINE_FACTORS`**——合成池仍是原来的 10 个，
以免把「加因子」与既有结论混在一起。

### 新接入的数据字段

`vwap.csv` / `vwap_adj.csv` 于 2026-08-18 加入 `DATABASE_FIELD_MAP`
（此前 CLAUDE.md 标注「本框架未使用」）。至此 Choice 的 29 个字段中，
框架已使用 22 个；未用的是 `xst`（st 已含）、`ev2` 的部分衍生等。

### 多因子合成（10 个因子）

**新口径（2026-10-02，2007-01 ~ 2026-09，组建日规则 a–g + 收盘到收盘）**：

| 方案 | LS 年化 | LS 夏普 | IC 均值 | ICIR | 月均股票数 |
|---|---|---|---|---|---|
| 等权 | +19.28% | 0.982 | 0.1024 | 0.637 | 1,990 |
| 滚动 ICIR 加权 | +17.77% | 1.012 | — | — | 1,990 |

逐条加入规则的中间结果见方法文档 §九。

**旧口径（2026-08，2007-01 ~ 2026-07，保留作对照）**：

| 方案 | LS 年化 | LS 夏普 | IC 均值 | ICIR | 月度胜率 |
|---|---|---|---|---|---|
| 等权 | +24.15% | 1.349 | 0.1104 | 0.762 | 69.8% |
| 滚动 ICIR 加权 | +24.77% | 1.379 | 0.1097 | 0.752 | 71.6% |

单因子 IC 最高 0.098，合成后到 0.11、ICIR 0.77——因子间互补性良好。

### 港股（3 个因子）

`amihud` (IC 0.0198) / `turnover_20` (-0.0274) / `cs_spread` (-0.0543)，
合成 LS 年化 6.55%、夏普 0.318。港股数据仍来自 iFinD，本轮迁移未动。

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

---

## 算力预算：默认不再吃满所有核（2026-08-21）

原先 `src/ml/models.py` 的 RF / LGBM 写死 `n_jobs=-1`，HistGBR 的 OpenMP 与
numpy 背后的 BLAS 也默认开满逻辑核，MacBook 连跑几十分钟会烫手并触发热降频。
现由 `src/config/compute.py` 统一管理：

- **默认** = 逻辑核的一半（本机 18 核 → **9 线程**）。
- **调节**：`--jobs N` / `--jobs all`（旧行为）/ `QUANT_JOBS=N`（对所有入口生效）。
- **散热间歇**：`python -m src.ml.run --cooldown 10`，每个滚动窗口之间停 10 秒。
  限线程压的是峰值功耗，间歇压的是持续积热，两者正交、可叠加。

环境变量那一层由 `src/__init__.py` 在 numpy 导入**之前**设好（这是 OMP_NUM_THREADS
唯一生效的时机），运行时再由 threadpoolctl 兜底。

**代价只落在 ML 训练，日常回测不收费。** 两条路径的线程敏感度完全不同，
不要把其中一条的结论外推到另一条：

| 路径 | 实测 | 结论 |
|---|---|---|
| `src.ml.run`（RF 300 树 / 20 万 × 10） | 18/12/9/6 线程 → 8.0/11.0/14.8/22.1s | 近似线性，默认的一半线程约合 **1.85 倍时长** |
| `src.main`（`ivol`，全库最 BLAS 密集的因子） | 18/9/4 线程 → 10.0/9.9/9.5s | **平坦，对线程数不敏感** |

`ivol` 走 `_rolling_ff3_ols` 的批量 OLS（226 月 × 5818 股，76 万非空格），
本以为是 BLAS 瓶颈，实测不是——它受内存带宽与 pandas 开销主导，加线程没用、
减线程也不疼。所以 `python -m src.main` 几乎不为线程预算付费；
真正吃满核心、也真正发热的是 ML 的 RF / LGBM。

异构核**没有**惩罚：RF 那组 6→18 线程近似线性，说明 E 核实实在在出力
（树之间无同步屏障）。故默认取「一半」而非「不超过性能核数」。

⚠️ 只影响速度与温度，**不改变任何结果**：树模型都固定了 `random_state`
（RF 四组的 `predict` 输出逐位相同），`ivol` 三组输出经
`np.array_equal(..., equal_nan=True)` 验证逐位相同，窗口之间本就无状态。

---

## 性能与已修复的 bug（2026-08-11）

全量回测 **27 分钟 → 3 分 48 秒**。四处「日期 × 股票」双重循环改为向量化，
均经离线对拍验证数学等价（详见 `Database/docs/【主文档】…` §6.1e）：

| 位置 | 原实现 | 改法 | 对拍 |
|---|---|---|---|
| `microstructure.py` `ivol`+`ff3_betas` | 248 月末 × ~3500 股票 × sklearn，约 174 万次拟合 | `_rolling_ff3_ols()` 一次批量 OLS，两因子共用 | 8.98e-16 |
| `microstructure.py` `roll_spread` | 260 月 × 5855 列 × `np.cov` | `ffill().shift(1)` 复现 dropna 相邻配对 | 6.36e-15 |
| `base.py` `neutralize_by_size` | 逐日期 sklearn | 闭式解 `β=cov/var` | 8.88e-16 |
| `base.py` 另两个中性化 | HDF5 在双重循环内反复读盘 | 循环外预加载 + `lstsq` | 3.82e-14 |
| `combine_factors.py` | `np.where(c, a/b, nan)` 触发除零警告 | `np.divide(where=)` | 输出逐格相同 |

`base.py` 已移除 sklearn 依赖。

### 第五处向量化：`calc_ic`（2026-08-24）

2026-08-11 那轮**漏掉了 `metrics.py` 的 `calc_ic`**。给 `main.py` 加阶段计时后
才暴露出来——它一个人占掉整轮回测的 **95.4%**：

```
改前（合计 1365.6s）              改后（合计 92.6s）
  calc_ic        1303.3s  95.4%     group_return     51.0s  55.0%
  group_return     49.1s   3.6%     calc_ic          28.5s  30.7%
  因子加载/计算      7.9s   0.6%     因子加载/计算      8.0s   8.6%
  plot_nav_curve    4.7s   0.3%     plot_nav_curve    4.7s   5.1%
```

**单因子段 1365.6s → 92.6s（14.7×），`calc_ic` 本身 46×。**

根因不是 IC 算法。原实现按 date 循环 234 次，每次做 4 次基于标签的 pandas 索引
（`.loc[date, common_cols]` + 三次 `reindex`/`dropna`），`common_cols` 有 5867 个
标签。`scipy.spearmanr` 在 5000 个元素上是毫秒级的，开销全在这 936 次宽表标签索引。

改法：循环外一次性对齐，然后按行矩阵运算。秩相关 = 秩上的 Pearson，是定义本身
而非近似。**关键是掩码要先于排秩**——原实现逐月把两边都非空的股票挑出来、
再在存活子集上排秩；若先排秩后掩码，秩的分母会变。

⚠️ **独立基准只有 12×，实跑却有 46×**，差异本身有信息量：原实现的瓶颈是宽表
标签索引，对内存压力高度敏感（实跑时 `results` 字典压着约 680MB，单次调用就从
5.2s 劣化到 21s）；新实现不做标签索引，基本不受影响。**所以脱离真实内存环境
测出来的加速比会严重低估收益。**

三层验证均通过：

| 验证 | 结果 |
|---|---|
| 62 因子 × spearman/pearson 离线对拍 | 最大绝对差 **3.14e-15**，NaN 位置完全一致 |
| `-W error::RuntimeWarning` 下重跑 | 零警告（`np.nanmean` 遇全 NaN 行的告警已消除） |
| 全流程 64 个 `*_ic_series.csv` 比对 | 最大绝对差 **1.67e-16** |

新瓶颈是 `group_return`（51s / 55%），但 92 秒的总耗时不值得再优化。

### 修复：IVOL 的截距重复相加

```python
# 原代码
residuals = y - (model.intercept_ + model.predict(X))
#                 ↑ 截距            ↑ predict 已含截距
```

`model.predict(X) = intercept_ + X @ coef_`，再加一次 `intercept_` 即重复。
后果是残差整体偏移 `-alpha`，`sse` 多出 `n·alpha²`，**IVOL 系统性偏大约 0.28%**。
对因子结论影响极小（IC 均值 -0.0660 → -0.0659）。

### 验收基准

`turnover_20_neutral` 跨越不同时间区间（2014~2021 → 2007~2026）与不同数据源
（Wind 存档 → Choice）仍逐项复现：

| | 旧结论 | 新结果 |
|---|---|---|
| IC 均值 / ICIR | -0.097 / -0.72 | -0.0980 / -0.6943 |
| G1 年化 / Sharpe | +22.5% / 0.76 | +23.32% / 0.7756 |
| G5 年化 / Sharpe | -2.1% / -0.06 | -2.53% / -0.0707 |

> 注：该基准对应的是**中性化版本**，不是 `turnover_20`（无中性化版新结果为 IC -0.0699）。

### 缓存对拍

`scripts/compare_cache.py` 比对 `output/cache/` 与 `output/cache_before_opt/`，
逐格核对因子矩阵，是「只变快、没变结果」这一声称的证据。

---

## 待办

| # | 事项 | 说明 |
|---|---|---|
| ~~1~~ | ~~`dividend_ratio` 换 Choice `LASTESTDIVIDEND`~~ | ✅ **已完成 2026-08-17**，见上节 |
| ~~2~~ | ~~`pe_ttm` 换 Choice `PETTM`~~ | ✅ **已完成 2026-08-17**，见下方警示 |
| ~~4~~ | ~~`pe1` 何去何从~~ | ✅ **已完成 2026-08-17**：改用 Choice `PE`，Datayes 退场 |
| ~~5~~ | ~~新增 `est_pe_ftm` 因子~~ | ✅ **已完成 2026-08-17**，连同 PS / EV/EBITDA / PEG / EV2 共 5 个 |
| ~~3~~ | ~~`pb` 换 Choice `PB`~~ | ✅ **已完成 2026-08-17** |
| 6 | `BACKTEST_START` 是否改 2005 | 数据已支持，但 2005~2006 A 股仅 1300~1400 只、股改期间大量停牌 |

### 成果记录（2026-09-30 登记）

成果分散在四处、新旧不一，没有一份汇总：

| # | 事项 | 说明 |
|---|---|---|
| 7 | 汇总成果记录 | 新建 `docs/` 下一份：单因子 + 合成 + ML 两轮 + 待办，作为唯一权威口径 |
| 8 | ~~ML 第 2 轮补分析~~ ⤴ 被 #19 新口径重跑取代（方法文档 §十七） | 2026-09-03 的 61 vs 109 特征（+48 会计）只有 `output/ml/compare.out` 原始数字，无解读 |
| 9 | ML 第 2 轮补跑 PLS / ENet | 第 1 轮冠军（夏普 2.33 / 2.04）没进第 2 轮，不知道加会计特征后如何 |
| 10 | ~~查 LGBM 夏普变化原因~~ ⤴ 旧口径问题，新口径重跑后不再追查 | 第 1 轮（08-20）1.47 → 第 2 轮（09-03）61 特征 1.83，记录里没写配置差异 |
| 11 | `ml_vs_ic_analysis.md` 遗留三项 | ① `--full` 复核树模型（第 1 轮是 `--fast`）② PLS/ENet 各做置零法重要性与张成检验（当时默认取了 LGBM/OLS-H）③ G1/G10 小盘股真实容量 |
| 12 | `reports/` 过时 | `A股量化因子回测报告.html`（06-30）、`…研究报告.pptx`（04-24）早于 Choice 迁移，更新或标作废 |
| ~~13~~ | ~~顶层 `MyProjects/CLAUDE.md` 数字过时~~ | ✅ 2026-10-02 已改为指向本文件与方法文档 |

### 回测口径对齐后续（2026-10-02 登记）

| # | 事项 | 说明 |
|---|---|---|
| 18 | ~~全部 62 个因子按新口径重跑~~ ✅ 2026-10-03：113 个因子（含会计块 48）已重跑，见方法文档 §十六 | 只重算了合成池 10 因子；「因子清单与状态」各表仍为旧口径。跑 `python -m src.main`（数据已更新，缓存会失效重算） |
| 19 | ~~ML 两轮按新口径重跑~~ ✅ 2026-10-03：见方法文档 §十七（roll_spread 58% 已复核：重要性口径伪象、非泄漏，见 §十七补充；遗留：OLS-H R² 失控、重要性改 IC 口径待定） | 训练标签（`src/ml/dataset.py`）与评估股票池都已改变，`output/ml/` 全部基于旧口径 |
| 20 | `size` 因子口径 | 仍用 `market_value`（含 H / B 股）；与规则 g 的 A 股市值不一致，属因子定义问题 |
| 21 | `ps_liq_beta` 估计输入 | 用了组建日筛选后的收益，理应用全部可得收益 |
| 22 | 合成池是否调整 | 剔最小 30% 后 `size`、`amihud` 大幅变弱、`pe1` 变强——合成池（原 10 个）是否重选，需单独讨论 |

### 2026-09-29 遗留

| # | 事项 | 说明 |
|---|---|---|
| 14 | `turnover_20_ff(_neutral)` 开关 | 已在 `FACTOR_FLAGS` 开启；替换无增益（ICIR -0.659 vs -0.669），决定留作对照还是关闭。核验见 `Database/docs/【核验】聚源自由流通换手率.md` |
| 15 | `MIN_ROLLING_VALID_DAYS` 实际不生效 | `_calc_turnover_20_raw` 的 `rolling(20).mean()` 默认 `min_periods=20`，窗口内一天 NaN 即整窗 NaN，门槛 10 形同虚设。改了会影响所有换手因子 |
| 16 | 毛利率研究续 | 初测 ICIR 0.07（中性化）。待做：分时段 IC、Δ毛利率、毛利率稳定性、作 ML 特征。脚本 `scripts/test_gross_margin.py` |
| 17 | `scripts/` 两个检验脚本未提交 | `test_gross_margin.py`、`compare_turnover_ff.py`，定稿后决定是否入库 |

✅ `data/stock/A/` 现已是 **100% Choice**（29/29 字段），Datayes 彻底退出。

## 排错备忘

**`force_recalc` 什么时候必须置 True**
- Database 数据更新（新增交易日）
- 数据源切换（如本轮 iFinD/Datayes → Choice，`close_adj` 复权基准变了）
- 修改了因子计算逻辑或回测区间

**`ff3_monthly.csv` 最后一行可能是不完整月份**
当月未结束时，末行由该月已有的少数交易日合成（例：2026-08-31 只由 8/3~8/5 三天算出）。
`BACKTEST_END` 自动取上一完整月末，回测不会用到它；但直接读该文件时需留意。

**`ff3_daily.csv` 是派生产物，不是下载数据**
由 `python -m src.ff3_builder` 生成，读 `data/stock/A/` 六个字段 + `macro/bond_yield_1y.csv`。
其中 `Rf = bond_yield_1y / 100 / 252`（5243 点逐点吻合，最大差 1e-17）。
Database 数据更新后需重跑 ff3_builder，再跑 main。
