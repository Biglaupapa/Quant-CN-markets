# 因子构建方法文档

本文件记录 QuantFramework 中每个因子的构建逻辑、关键参数、数据来源及重要设计决策。
所有设计决策均附文献依据或经验判断说明，供后续回顾和修改参考。

---

## 全局设置

### 回测起点

**当前设置**：`BACKTEST_START = "2007-01-01"`

选择 2007 年起：2007 年前 A 股市场规模较小，PB/PE 等基本面数据可信度偏低，
各因子统一对齐至此起点，使单因子与多因子合成结果具有可比性。

### 价格复权开关（`USE_ADJ_PRICE`）

**位置**：`config/settings.py`

```python
USE_ADJ_PRICE = True   # True = 后复权；False = 不复权
```

影响以下因子的价格字段选择：`reversal_20`、`momentum_12_1`、`amihud_zero_adj`、
`roll_spread`、`overnight_ret`、`volatility_30`。

**不受此开关影响的因子**（原因见各因子说明）：
- `amihud`：日度收益率计算必须使用后复权价，否则除权日产生虚假大收益
- `cs_spread`：跨日高低价比较必须使用复权价（见 cs_spread 说明）
- `turnover_20` / `turnover_20_neutral`：仅使用 `turn` 字段，不涉及价格

### 中性化变量说明

市值中性化统一使用 **log(流通市值)**，数据源为 `neg_market_value`（Datayes，2004+）。
不再使用 `close × float_shares` 估算（该方式为历史遗留，已替换）。

---

## 微观结构因子（`factors/microstructure.py`）

---

### 1. reversal_20 — 短期反转

**计算公式**：

$$\text{Reversal}_{i,t} = \frac{P_{i,t}}{P_{i,t-20}} - 1$$

过去 20 个交易日的累计收益率，月末取值。

**因子方向**：负向（G1=过去最跌 → 未来反弹；G5=过去最涨 → 未来回调）

**数据**：`close_adj`（或 `close`，由 `USE_ADJ_PRICE` 控制）

**参数**：`REVERSAL_WINDOW = 20`（交易日）

**中性化**：无

**回测结果（2004-2026）**：
- IC均值 = -0.066，ICIR = -0.44，|IC|>0.02占比 = 89%
- G1年化+16.6%，G5年化+0.4%，单调性较好但 G1/G2 差距小

---

### 2. momentum_12_1 — 中期动量

**计算公式**（月度）：

$$\text{MOM}_{i,t} = \frac{P_{i,t-1}}{P_{i,t-12}} - 1$$

跳过最近 1 个月（`t-1` 到 `t`），避免短期反转干扰，使用 12-1 个月累计收益。

**因子方向**：理论正向（强者恒强），但 A 股实证失效

**数据**：`close_adj`（月度）

**参数**：`MOMENTUM_LONG = 12`，`MOMENTUM_SKIP = 1`

**中性化**：无

**回测结果（2004-2026）**：
- IC均值 = -0.014，ICIR = -0.08，接近零，**因子无效**
- G1-G5 呈倒 U 型（G3 最高），分组无单调性

**A股特殊性**：A 股以散户为主，追涨杀跌盛行，过去 12 个月涨幅最大的股票
（G5）往往已被过度定价，后续反转而非延续。与海外动量效应相反，
是 A 股定价机制的结构性特征，不是计算错误。

---

### 3. turnover_20 — 换手率（无中性化）

**计算公式**：

$$\text{TURN}_{i,t} = \frac{1}{20} \sum_{d=t-19}^{t} \text{turn}_{i,d}$$

过去 20 个交易日的平均换手率，月末取值。

**因子方向**：负向（低换手率 → 高未来收益）

**数据**：`turn`

**参数**：`TURNOVER_WINDOW = 20`

**中性化**：无（原始信号）

**回测结果（2004-2026）**：
- IC均值 = -0.096，**ICIR = -0.72**，|IC|>0.02占比 = 89%
- G1年化+23.6%（Sharpe=0.79），G5年化-1.3%，**G1-G5单调递减完美**
- **最强因子**，信号最稳定

**经济逻辑**：低换手代表机构持仓稳定、散户炒作少，筹码结构优良。

---

### 3b. turnover_20_neutral — 换手率（流通市值中性化）

同 `turnover_20`，但在截面上对 log(流通市值) 回归取残差，剔除换手率与规模的线性相关。

**数据**：`turn` + `neg_market_value`（中性化变量）

**中性化**：log(流通市值)，`neg_market_value`（Datayes，2004+）

**用途**：与 `turnover_20` 对比，判断市值中性化是否提升因子质量。

---

### 4. amihud — Amihud 非流动性（无中性化）

**计算公式**（Amihud, 2002）：

$$\text{ILLIQ}_{i,t} = \frac{1}{D} \sum_{d=1}^{D} \frac{|R_{i,d}|}{\text{Amt}_{i,d} / 10^6} \times \text{ILLIQ\_SCALE}$$

其中 $D$ 为过去 3 个月的交易日数，$\text{Amt}$ 为成交额（元），除以 $10^6$ 换算为百万元，
`ILLIQ_SCALE = 1e5` 为缩放系数，使因子数值在合理范围内。

**滞后处理**：因子整体滞后 `AMIHUD_LAG_MONTHS = 1` 个月，避免月末数据前瞻偏差。

**因子方向**：正向（ILLIQ 越大 = 流动性越差 = G5；G5 未来收益越高）

**数据**：`close_adj`（后复权，计算日度收益率必须复权）、`amt`（成交额，元）

**参数**：`AMIHUD_WINDOW_MONTHS = 3`，`AMIHUD_LAG_MONTHS = 1`，`ILLIQ_SCALE = 1e5`

**中性化**：无（原始信号）

**回测结果（2004-2026）**：
- IC均值 = +0.049，ICIR = +0.31，|IC|>0.02占比 = 91%
- G1年化+5.1%，G5年化+22.8%，**G1-G5单调递增完美**
- LS年化+16.6%，**LS最大回撤仅-24.9%**（全部因子中最稳健）

**注意**：`close_adj` 不受 `USE_ADJ_PRICE` 控制，始终使用后复权价。

---

### 4b. amihud_neutral — Amihud 非流动性（流通市值中性化）

同 `amihud`，但截面回归剔除与 log(流通市值) 的线性相关后取残差。

**数据**：`close_adj`、`amt` + `neg_market_value`（中性化变量）

**中性化**：log(流通市值)，`neg_market_value`（Datayes，2004+）

**用途**：与 `amihud` 对比，判断剥离规模效应后流动性信号的纯度。

---

### 5. amihud_zero_adj — Amihud 零交易日调整版（无中性化）

**计算公式**（IREF-sentiment 版本）：

$$\text{ILLIQ\_ZA}_{i,t} = \ln\!\left(\text{mean}\!\left(\frac{|R_{i,d}|}{\text{Vol}_{i,d}}\right)\right) \times (NT_t + 1)$$

其中 $NT_t$ 为当月零交易日（成交量为 0）占比，对零流动性日期施加更大惩罚。
使用成交量（股数）而非成交额。

**因子方向**：正向（同 amihud）

**数据**：`close_adj`、`volume`（成交量）

**中性化**：无（原始信号）

**回测结果（2004-2026）**：
- IC均值 = +0.044，ICIR = +0.27，|IC|>0.02占比 = 92%（全部因子最高）
- LS年化+14.0%，LS最大回撤-30.4%

**与 amihud 关系**：方向相同，强度略弱，但信号频率更稳定。两者相关性较高，
实际使用时选一个即可，amihud 在 IC 强度上更优。

---

### 5b. amihud_zero_adj_neutral — Amihud 零交易日调整版（流通市值中性化）

同 `amihud_zero_adj`，但截面回归剔除与 log(流通市值) 的线性相关后取残差。

**数据**：`close_adj`、`volume` + `neg_market_value`（中性化变量）

**中性化**：log(流通市值)，`neg_market_value`（Datayes，2004+）

---

### 6. cs_spread — Corwin-Schultz 高低价价差

**计算公式**（Corwin & Schultz, 2012）：

$$\beta = (\ln H_t/L_t)^2 + (\ln H_{t+1}/L_{t+1})^2, \quad
  \gamma = (\ln \max(H_t,H_{t+1}) / \min(L_t,L_{t+1}))^2$$

$$\alpha = \frac{\sqrt{2\beta}-\sqrt{\beta}}{3-2\sqrt{2}} - \sqrt{\frac{\gamma}{3-2\sqrt{2}}}, \quad
  \text{Spread} = \frac{2(e^\alpha - 1)}{1+e^\alpha}$$

月度均值，要求至少 `MIN_ROLLING_VALID_DAYS` 个有效值。

**负值处理：截断为 0**（Corwin & Schultz 原文建议，原因见下）

当 2 日波动率远大于 1 日波动率的 2 倍时，α 变负，导致 spread 估计值为负数。
CS Spread 的负值出现在**输出端（spread 值本身）**，bid-ask spread 经济含义天然 ≥ 0，
负值只是采样噪音的产物，截断为 0 是合理的下界修正。

**必须使用复权高低价**：除权日前后不复权的高低价差异可达数倍，导致跨日价差估计严重失真。
故 `high_adj`/`low_adj` **不受 `USE_ADJ_PRICE` 控制**，始终使用复权价。

**因子方向**：负向（G1=价差最小/最流动 → 未来收益最高）

**数据**：`high_adj`、`low_adj`

**中性化**：无

**回测结果（2004-2026）**：
- IC均值 = -0.057，ICIR = -0.46，|IC|>0.02占比 = 86%
- G1年化+15.5%，G5年化+4.8%

---

### 7. roll_spread — Roll 价差

**计算公式**（Roll, 1984）：

$$S_{i,t} = 2\sqrt{-\text{Cov}(r_{i,d}, r_{i,d-1})} \quad \text{当 } \text{Cov} < 0$$

**核心设计决策：Cov ≥ 0 时置 NaN（非 0）**

Roll 方法的理论前提是 bid-ask bounce 产生负序列自相关。Cov ≥ 0 直接违反此前提，
不是"估计出了负的价差"，而是"连判断价差是否存在的能力都没有了"——置 NaN，非 0。
（置 0 会导致约 57% 的数据同值，`pd.qcut` 无法形成有意义的 5 个分组。）

**因子方向**：负向（G1=Roll spread 最小/最流动 → 未来收益最高）

**数据**：`close_adj`（由 `USE_ADJ_PRICE` 控制）

**中性化**：无

**回测结果（2004-2026）**：
- IC均值 = -0.053，ICIR = -0.499
- G1年化+14.4%，G5年化+4.2%，单调性良好

---

### 8. overnight_ret — 隔夜收益率

**计算公式**：

$$\text{OVN}_{i,d} = \frac{\text{Open}_{i,d}}{\text{Close}_{i,d-1}} - 1$$

月内所有交易日隔夜收益率的均值。

**因子方向**：理论正向，A 股实证接近无效

**数据**：`open_adj`（或 `open`）、`close_adj`（或 `close`），由 `USE_ADJ_PRICE` 控制

**中性化**：无

**回测结果（2004-2026）**：
- IC均值 = +0.010，ICIR = +0.13，LS年化仅+0.8%，**基本无效**

---

### 9. volatility_30 — 短期波动率

**计算公式**：

$$\text{VOL}_{i,t} = \text{std}(r_{i,d})_{d \in [t-29, t]}$$

过去 30 个交易日日度收益率的标准差，月末取值。

**因子方向**：负向（低波动率 → 高未来收益）—— 低波动异象

**数据**：`close_adj`（由 `USE_ADJ_PRICE` 控制）

**参数**：`VOLATILITY_WINDOW = 30`

**中性化**：无

**回测结果（2004-2026）**：
- IC均值 = -0.064，ICIR = -0.37，|IC|>0.02占比 = 89%
- G1年化+14.2%（Sharpe=0.52），G5年化+3.7%

---

## 基本面因子（`factors/fundamental.py`）

**数据来源**：Database `data/stock/A/`（Datayes，覆盖 2004-至今）

**中性化变量**：`neg_market_value`（流通市值），`FactorLoading_Industry_arch.h5`（行业哑变量）

---

### 10. pb — 市净率

**计算公式**：月末截面 PB 值，越小代表越"价值"。

**数据**：`pb.csv`（Database，2004+）

**中性化**：无（原始估值信号）

**因子方向**：负向（低 PB = 价值股，预期收益更高）

---

### 11. bm — 账面市值比（Book-to-Market）

**计算公式**：

$$\text{B/M}_{i,t} = \log\!\left(\frac{1}{\text{PB}_{i,t}}\right) = -\log(\text{PB}_{i,t})$$

参考 Fama-French HML 构建逻辑，使用 log 变换后的 B/M 而非原始 PB。

**数据**：`pb.csv`（Database，2004+），PB ≤ 0 置 NaN

**中性化**：无

**因子方向**：正向（B/M 越高 = 价值股 = 正向信号）

**与 pb 的关系**：方向相反（pb 负向，bm 正向），变换后的 bm 分布更接近正态，
理论上更适合截面回归。两者可对比单因子 IC，选择更优的版本用于合成。

---

### 12. pe_ttm — 市盈率 TTM（静态）

**字段定义**：**滚动市盈率（PE TTM）** = 总市值 / 归属于母公司所有者的净利润 TTM

其中净利润 TTM 是过去连续四个季度净利润之和（即滚动12个月累计净利润，
基于**已披露**的最新年报或半年报，属于静态的历史会计数据）。

**计算公式**：月末截面值，负值（亏损公司）置 NaN。

**数据**：`pe_ttm.csv`（Database，Datayes getMktEqud.PE，2004+）

**中性化**：无

**因子方向**：负向（低 PE TTM = 价值股）

**回测结果（2007-2026）**：
- IC均值 = -0.037，ICIR = -0.241，信号强度中等偏弱
- 不如 pe1（动态）灵敏，原因见下方 pe1 说明

---

### 13. pe1 — 动态市盈率

**字段定义**：**动态市盈率（PE1）** = 总市值 / 归属于母公司所有者的净利润（最新一期财报年化）

其中"最新一期财报年化"是以最新披露的季报/半年报/年报的单期净利润乘以年化系数推算，
而非四季度累计。例如：最新季报净利润 × 4 = 年化净利润。

**计算公式**：月末截面值，负值（亏损公司）置 NaN。

**与 pe_ttm 的核心差异**：

| 指标 | 分子 | 信息来源 | 及时性 |
|------|------|---------|--------|
| `pe_ttm` | 净利润 TTM（滚动12个月） | 已披露的最近4个季度累计 | 较滞后 |
| `pe1`    | 净利润（最新一期年化）  | 最新单季报年化          | 更及时 |

**数据**：`pe1.csv`（Database，Datayes getMktEqud.PE1，2004+）

**中性化**：无

**因子方向**：负向（低 PE1 = 估值低）

**回测结果（2007-2026）**：
- IC均值 = -0.046，ICIR = -0.298，优于 pe_ttm（ICIR -0.241）
- pe1 对盈利变化反应更快，能更早捕捉盈利改善/恶化的估值信号

**选择依据**：同类因子对比（pe1 vs pe_ttm），pe1 在 ICIR 和 LS 收益上均优于 pe_ttm，
且经济逻辑上更及时，因此选用 pe1 纳入合成，pe_ttm 作为对照保留在 FACTOR_FLAGS 中。

---

### 14. dividend_yield — 股息率 TTM

**计算公式**：月末截面股息率（近12个月），越高代表股息越丰厚。

**数据**：`dividend_ratio.csv`（Database，2004+）

**中性化**：无

**因子方向**：正向（高股息 = 好）

---

### 15. size — 总市值因子

**计算公式**：$\text{SIZE} = \ln(\text{market\_value})$

**数据**：`market_value`（Datayes getMktDivYield，2004+）

**中性化**：无（市值因子本身是中性化的控制变量，不宜自我中性化）

**因子方向**：负向（小市值溢价）

---

### 16. size2 — 流通市值因子

**计算公式**：$\text{SIZE2} = \ln(\text{neg\_market\_value})$

与 `size`（总市值）的差异体现在限售股比例较高的个股上。

**数据**：`neg_market_value`（Datayes getMktEqud，2004+）

**中性化**：无

**因子方向**：负向（小流通市值溢价）

---

### 17. net_profit_yoy — 净利润同比增速

**计算公式**：$\text{YoY}_t = \frac{NP_t}{NP_{t-4}} - 1$（季度同比）

**滞后处理**：滞后 3 个季度（财报披露延迟，`NP_YOY_LAG_QUARTERS = 3`）

**数据来源**：仅存档 `_archive/raw_data/归属母公司净利润.csv`（2014-2021）

**中性化**：log(流通市值) + 行业双重中性化

**因子方向**：正向（净利润增速越高越好）

**当前状态**：唯一仍依赖存档的因子，回测区间受限于存档时间范围（2014-2021）

---

## 待激活因子（需额外数据）

| 因子 | 所需数据 | 说明 |
|------|---------|------|
| `ps_gamma` | `marketrtn_daily.csv` | Pastor-Stambaugh Gamma |
| `ps_liq_beta` | ps_gamma + 市场收益率 | PS 流动性 Beta（36月滚动） |
| `ap_betas` | 日度流通市值序列 | Acharya-Pedersen β1-β5 |
| `capm_beta` | `marketrtn_daily.csv` | CAPM Beta（240日滚动） |
| `ivol` | FF3日度因子 + 无风险利率 | 特质波动率 |
| `ff3_betas` | FF3日度因子 + 无风险利率 | FF3三因子Beta |

---

## 已验证结果汇总（2007-01 ~ 2026-03，231个月，6077只股票）

### 单因子回测结果

| 因子 | IC均值 | ICIR | LS年化 | LS Sharpe | 有效性 | 进入合成？ |
|------|--------|------|--------|-----------|--------|----------|
| turnover_20 | -0.097 | **-0.715** | +21.4% | 1.36 | ★★★ 强 | ✅ |
| turnover_20_neutral | -0.098 | -0.718 | +21.5% | 1.38 | ★★★（≈同上）| ❌ 二选一→选无中性化 |
| roll_spread | -0.053 | -0.490 | — | 0.71 | ★★★ | ✅ |
| reversal_20 | -0.064 | -0.450 | +14.1% | 0.99 | ★★★ | ✅ |
| cs_spread | -0.056 | -0.447 | — | 0.60 | ★★★ | ✅ |
| amihud | +0.059 | +0.387 | +17.3% | 1.20 | ★★★ | ✅ |
| volatility_30 | -0.066 | -0.383 | — | 0.44 | ★★ | ✅ |
| size | -0.066 | **-0.355** | +21.2% | 1.16 | ★★★ | ✅ 基本面 |
| bm | +0.055 | +0.334 | — | 0.49 | ★★ | ✅ 基本面 |
| pe1 | -0.046 | -0.298 | — | 0.55 | ★★ | ✅ 基本面 |
| amihud_zero_adj | +0.044 | +0.261 | +12.8% | 0.75 | ★★ | ⚠️ 见注1 |
| amihud_neutral | +0.025 | +0.213 | — | 0.51 | ★（弱化）| ❌ 二选一→选无中性化 |
| overnight_ret | +0.017 | +0.222 | — | 0.33 | ★ 弱 | ⚠️ 可选 |
| pe_ttm | -0.037 | -0.241 | — | 0.37 | ★ | ❌ 二选一→选pe1 |
| size2 | -0.049 | -0.263 | — | 0.93 | ★★ | ❌ 二选一→选size |
| dividend_yield | +0.020 | +0.181 | — | 1.91 | ⚠️ 异常 | ❌ 见注2 |
| pb | -0.055 | -0.333 | — | 0.62 | ★★ | ❌ 二选一→选bm |
| amihud_zero_adj_neutral | +0.022 | +0.153 | — | 0.33 | ★（弱化）| ❌ |
| momentum_12_1 | -0.020 | -0.128 | — | 0.31 | ✗ 无效 | ❌ A股失效 |

**注1 — amihud vs amihud_zero_adj**：
月度截面 Spearman 相关性 ρ = 0.694（2007-2026均值），未超过 0.8 的排除阈值。
amihud（ICIR 0.387）明显强于 amihud_zero_adj（ICIR 0.261）。
默认合成池只纳入 amihud；若研究需要流动性信号多样性可加回。

**注2 — dividend_yield 异常**：
LS Sharpe = 1.91 远高于 ICIR = 0.181，两者严重不匹配。
原因诊断：65.3% 的股票因子值 < 0（股息率低于截面均值，含大量零股息股票）。
G1（低股息/零股息）= 小市值成长股，G5（高股息）= 银行/央企/公用事业。
LS 收益大概率混入规模溢价，并非纯股息信号。建议中性化市值后重检，暂不纳入合成。

---

### 因子相关性矩阵关键发现（2007-2026，月度截面 Spearman 均值）

高相关对（|ρ| > 0.5）：

| 因子对 | ρ | 说明 |
|--------|---|------|
| amihud ↔ size | -0.770 | **最高相关**：小市值股票天然非流动，两者捕获相似信息。同时纳入合成时要注意冗余。 |
| amihud_zero_adj ↔ size | -0.530 | 同上模式，强度稍低 |
| amihud ↔ amihud_zero_adj | +0.694 | 两种 Amihud 变体高度重叠，选一即可 |
| cs_spread ↔ volatility_30 | +0.542 | 两者均与价格波动相关（高低价差 vs 日度收益率 std） |
| turnover_20 ↔ volatility_30 | +0.505 | 高换手率伴随高波动率（散户主导效应） |
| pe1 ↔ dividend_yield | -0.505 | 高PE（成长股）→ 低股息率，符合预期 |

热力图路径：`output/img/factor_correlation.png`

---

## 参考文献

- Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*.
- Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance*.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*.
- Fong, K. Y. L., Holden, C. W., & Trzcinka, C. A. (2017). What are the best liquidity proxies for global research? *Review of Finance*.
- Goyenko, R. Y., Holden, C. W., & Trzcinka, C. A. (2009). Do liquidity measures measure liquidity? *Journal of Financial Economics*.
- Pastor, L., & Stambaugh, R. F. (2003). Liquidity risk and expected stock returns. *Journal of Political Economy*.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance*.
