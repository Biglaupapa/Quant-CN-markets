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

### 12. pe_ttm — 市盈率 TTM

**计算公式**：月末截面 PE（TTM），负值（亏损公司）置 NaN。

**数据**：`pe_ttm.csv`（Database，2004+）

**中性化**：无

**因子方向**：负向（低 PE = 价值股）

---

### 13. pe1 — 动态市盈率

**计算公式**：月末截面 PE1（滚动12个月盈利），负值置 NaN。

PE1 与 pe_ttm 的区别：
- `pe_ttm`：总市值 / 最近一年报告期净利润（静态，基于已披露年报）
- `pe1`：总市值 / 滚动12个月盈利（动态，含最新季报，更及时）

**数据**：`pe1.csv`（Database，2004+）

**中性化**：无

**因子方向**：负向（低 PE1 = 估值低）

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

## 已验证结果汇总（2004-01 ~ 2026-03，267个月，6077只股票）

> 注：回测起点已改为 2007-01-01，下表结果为旧版（2004起），待重新运行后更新。

| 因子 | IC均值 | ICIR | LS年化 | LS最大回撤 | 有效性 |
|------|--------|------|--------|-----------|--------|
| turnover_20 | -0.096 | **-0.72** | -20.0% | -99.3% | **强有效** |
| cs_spread | -0.057 | -0.46 | -8.5% | -87.9% | 有效 |
| reversal_20 | -0.066 | -0.44 | -15.8% | -97.9% | 有效 |
| volatility_30 | -0.064 | -0.37 | -8.1% | -86.0% | 有效 |
| amihud | +0.049 | +0.31 | +16.6% | **-24.9%** | **有效且稳健** |
| amihud_zero_adj | +0.044 | +0.27 | +14.0% | -30.4% | 有效 |
| roll_spread | -0.053 | -0.499 | -8.46% | — | 有效 |
| overnight_ret | +0.010 | +0.13 | +0.8% | -43.8% | 无效 |
| momentum_12_1 | -0.014 | -0.08 | -2.9% | -65.8% | **无效**（A股动量反转） |

---

## 参考文献

- Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*.
- Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance*.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*.
- Fong, K. Y. L., Holden, C. W., & Trzcinka, C. A. (2017). What are the best liquidity proxies for global research? *Review of Finance*.
- Goyenko, R. Y., Holden, C. W., & Trzcinka, C. A. (2009). Do liquidity measures measure liquidity? *Journal of Financial Economics*.
- Pastor, L., & Stambaugh, R. F. (2003). Liquidity risk and expected stock returns. *Journal of Political Economy*.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance*.
