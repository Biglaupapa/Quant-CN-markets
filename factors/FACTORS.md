# 因子构建方法文档

本文件记录 QuantFramework 中每个因子的构建逻辑、关键参数、数据来源及重要设计决策。
所有设计决策均附文献依据或经验判断说明，供后续回顾和修改参考。

---

## 全局设置

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
- `turnover_20`：市值计算用不复权收盘价 × 流通股本（复权价会高估历史市值）

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

**回测结果（2004-2026）**：
- IC均值 = -0.066，ICIR = -0.44，|IC|>0.02占比 = 89%
- G1年化+16.6%，G5年化+0.4%，单调性较好但 G1/G2 差距小

**注意**：A股短期反转效应存在，但强度弱于换手率因子。

---

### 2. momentum_12_1 — 中期动量

**计算公式**（月度）：

$$\text{MOM}_{i,t} = \frac{P_{i,t-1}}{P_{i,t-12}} - 1$$

跳过最近 1 个月（`t-1` 到 `t`），避免短期反转干扰，使用 12-1 个月累计收益。

**因子方向**：理论正向（强者恒强），但 A 股实证失效

**数据**：`close_adj`（月度）

**参数**：`MOMENTUM_LONG = 12`，`MOMENTUM_SKIP = 1`

**回测结果（2004-2026）**：
- IC均值 = -0.014，ICIR = -0.08，接近零，**因子无效**
- G1-G5 呈倒 U 型（G3 最高），分组无单调性

**A股特殊性**：A 股以散户为主，追涨杀跌盛行，过去 12 个月涨幅最大的股票
（G5）往往已被过度定价，后续反转而非延续。与海外动量效应相反，
是 A 股定价机制的结构性特征，不是计算错误。

---

### 3. turnover_20 — 换手率（市值中性化）

**计算公式**：

$$\text{TURN}_{i,t} = \frac{1}{20} \sum_{d=t-19}^{t} \text{turn}_{i,d}$$

过去 20 个交易日的平均换手率，月末取值，随后做市值中性化。

**市值中性化**：以 $\log(\text{收盘价} \times \text{流通股本})$ 为控制变量，
对换手率截面回归取残差，剔除换手率与市值规模的线性相关部分。

**因子方向**：负向（低换手率 → 高未来收益）

**数据**：`turn`（换手率）、`close`（不复权，用于市值）、`float_shares`（流通股本）

**参数**：`TURNOVER_WINDOW = 20`，`TURN_NEUTRALIZE_SIZE = True`

**回测结果（2004-2026）**：
- IC均值 = -0.096，**ICIR = -0.72**，|IC|>0.02占比 = 89%
- G1年化+23.6%（Sharpe=0.79），G5年化-1.3%，**G1-G5单调递减完美**
- **最强因子**，信号最稳定

**经济逻辑**：低换手代表机构持仓稳定、散户炒作少，筹码结构优良。

---

### 4. amihud — Amihud 非流动性（版本A）

**计算公式**（Amihud, 2002）：

$$\text{ILLIQ}_{i,t} = \frac{1}{D} \sum_{d=1}^{D} \frac{|R_{i,d}|}{\text{Amt}_{i,d} / 10^6} \times \text{ILLIQ\_SCALE}$$

其中 $D$ 为过去 3 个月的交易日数，$\text{Amt}$ 为成交额（元），除以 $10^6$ 换算为百万元，
`ILLIQ_SCALE = 1e5` 为缩放系数，使因子数值在合理范围内。

**滞后处理**：因子整体滞后 `AMIHUD_LAG_MONTHS = 1` 个月，避免月末数据前瞻偏差。

**因子方向**：正向（ILLIQ 越大 = 流动性越差 = G5；G5 未来收益越高）

**数据**：`close_adj`（后复权，计算日度收益率必须复权）、`amt`（成交额，元）

**参数**：`AMIHUD_WINDOW_MONTHS = 3`，`AMIHUD_LAG_MONTHS = 1`，`ILLIQ_SCALE = 1e5`

**成交额单位说明**：Database 中 `amt` 单位为元（验证：平安银行日均约 10 亿元 = 1e9）。
计算时先除以 1e6 换算为百万元，再乘以 ILLIQ_SCALE=1e5，等效缩放系数为 0.1（元口径）。

**回测结果（2004-2026）**：
- IC均值 = +0.049，ICIR = +0.31，|IC|>0.02占比 = 91%
- G1年化+5.1%，G5年化+22.8%，**G1-G5单调递增完美**
- LS年化+16.6%，**LS最大回撤仅-24.9%**（全部因子中最稳健）

**注意**：`close_adj` 不受 `USE_ADJ_PRICE` 控制，始终使用后复权价。

---

### 5. amihud_zero_adj — Amihud 零交易日调整版（版本C）

**计算公式**（IREF-sentiment 版本）：

$$\text{ILLIQ\_ZA}_{i,t} = \ln\!\left(\text{mean}\!\left(\frac{|R_{i,d}|}{\text{Vol}_{i,d}}\right)\right) \times (NT_t + 1)$$

其中 $NT_t$ 为当月零交易日（成交量为 0）占比，对零流动性日期施加更大惩罚。
使用成交量（股数）而非成交额。

**因子方向**：正向（同 amihud）

**数据**：`close_adj`、`volume`（成交量）

**回测结果（2004-2026）**：
- IC均值 = +0.044，ICIR = +0.27，|IC|>0.02占比 = 92%（全部因子最高）
- LS年化+14.0%，LS最大回撤-30.4%

**与 amihud 关系**：方向相同，强度略弱，但信号频率更稳定。两者相关性较高，
实际使用时选一个即可，amihud 在 IC 强度上更优。

---

### 6. cs_spread — Corwin-Schultz 高低价价差

**计算公式**（Corwin & Schultz, 2012）：

$$\beta = (\ln H_t/L_t)^2 + (\ln H_{t+1}/L_{t+1})^2, \quad
  \gamma = (\ln \max(H_t,H_{t+1}) / \min(L_t,L_{t+1}))^2$$

$$\alpha = \frac{\sqrt{2\beta}-\sqrt{\beta}}{3-2\sqrt{2}} - \sqrt{\frac{\gamma}{3-2\sqrt{2}}}, \quad
  \text{Spread} = \frac{2(e^\alpha - 1)}{1+e^\alpha}$$

月度均值，要求至少 `MIN_ROLLING_VALID_DAYS` 个有效值。

**负值处理：截断为 0（Corwin & Schultz 原文建议）**

当 2 日波动率远大于 1 日波动率的 2 倍时，α 变负，导致 spread 估计值为负数。
CS Spread 的负值出现在**输出端（spread 值本身）**：公式在统计上产生了一个负的价差估计，
但 bid-ask spread 作为交易摩擦成本，其经济含义天然 ≥ 0，负值只是采样噪音的产物，
并非模型假设本身失效——公式的推导逻辑依然成立，只是这次估计偏低了。
因此将负值截断为 0（默认下界）是合理的：保留估计值的分布信息，仅修正方向错误的端点。

Corwin & Schultz 指出，月度均值即使包含部分被截为 0 的 2 日窗口，
仍然"几乎总是正的"，月度因子不受实质性影响（见 Goyenko et al., 2009 综述）。

**必须使用复权高低价**：CS Spread 的核心是计算相邻两天的跨日最高/最低价比值
$\max(H_t, H_{t+1}) / \min(L_t, L_{t+1})$。如果使用不复权价格，除权日前后的
$H_t$（含权）和 $H_{t+1}$（除权后）差异可达数倍（如五粮液 2023 年约 15.4 倍），
导致价差估计严重失真。故 `high_adj`/`low_adj` **不受 `USE_ADJ_PRICE` 控制**，始终使用复权价。

**轻微前瞻偏差**：公式需要 $t$ 和 $t+1$ 两天，月末最后一天借用了下月第一天数据
（约 1/22 = 4.5% 的影响），属于 Corwin-Schultz 方法的固有特性，文献中普遍接受。

**因子方向**：负向（G1=价差最小/最流动 → 未来收益最高）

**数据**：`high_adj`、`low_adj`

**回测结果（2004-2026）**：
- IC均值 = -0.057，ICIR = -0.46，|IC|>0.02占比 = 86%
- G1年化+15.5%，G5年化+4.8%

**与 Amihud 方向相反的解读**：两者均为流动性指标，但：
- Amihud（正向）：衡量价格冲击 / 市场深度，主要捕捉小盘规模效应
- CS Spread（负向）：衡量买卖摩擦成本，高价差股票往往是散户过度投机的热门股，
  长期因过度定价而跑输

---

### 7. roll_spread — Roll 价差

**计算公式**（Roll, 1984；Goyenko et al., 2009；Fong et al., 2017）：

$$S_{i,t} = 2\sqrt{-\text{Cov}(r_{i,d}, r_{i,d-1})} \quad \text{当 } \text{Cov} < 0$$

使用月内日度收益率的样本协方差。使用收益率（而非价格变动）隐式实现了价格标准化：

$$\text{Cov}(r_t, r_{t-1}) \approx \frac{\text{Cov}(\Delta P_t, \Delta P_{t-1})}{\bar{P}^2}$$

即等价于 Fong (2017) 的 $2\sqrt{-\text{Cov}(\Delta P)}/\bar{P}$，使跨股票比较具有可比性。

**核心设计决策：Cov ≥ 0 时置 NaN（非 0）**

**与 CS Spread 的关键区别**：两者都在度量 bid-ask spread，但负值/无效情形的来源不同：

| | CS Spread | Roll Spread |
|---|---|---|
| 问题出现位置 | **输出端**：spread 估计值本身为负 | **输入端**：中间量协方差 ≥ 0 |
| 原因 | 采样噪音导致估计偏低，公式逻辑仍成立 | 模型核心假设失效，公式无法运行 |
| 处理方式 | 截为 **0**（下界修正） | 置为 **NaN**（整个估计作废） |

Roll 方法的理论前提是：买卖价差的存在会造成成交价在买方价（bid）和卖方价（ask）
之间交替，产生负的序列自相关（bid-ask bounce）。**Cov ≥ 0 直接违反这一前提**——
不是"估计出了负的价差"，而是"连判断价差是否存在的能力都没有了"。
这与 CS Spread 中"公式仍有意义、只是估计值碰巧为负"有本质区别。

因此对 Roll Spread 而言，Cov ≥ 0 应置 NaN，而非 0。

文献惯例（Goyenko 2009, Fong 2017）将 Cov ≥ 0 赋值为 0，适用于**流动性水平的横向比较**，
即将这类股票视为"价差为零/完全流动"。但在截面因子回测中，0 值导致约 57% 的
股票-月份取同一标准化值，`pd.qcut` 无法形成 5 个有意义的分组
（实测 G5 只有 13 个有效月份/267 个月）。NaN 处理在因子排序场景中更恰当。

**经济逻辑**（Fong 2017 原文）：
> "a positive sample serial correlation is most likely to occur when the true,
> population value of the serial correlation is very small,
> which corresponds to a highly liquid stock."

Cov ≥ 0 意味着 bid-ask bounce 不可检测，Roll 方法对这类股票**没有估计能力**。
将其置 NaN 并非丢弃有用信息，而是承认"此方法在此处无效"。

在 Cov < 0 的子集（约 43% 的股票-月份）内部排序，经济含义清晰：
这些股票存在可检测的买卖价差，Roll spread 越大表示流动性越差。

**与 Amihud 的潜在重叠**：保留的 43% 子集（流动性较差股票）与 Amihud
高值股票存在较大重叠，两因子相关性可能较高，后续可验证独立信息含量。

**因子方向**：负向（G1=Roll spread 最小/最流动 → 未来收益最高）

**回测结果（2004-2026，NaN 修复后）**：
- IC均值 = -0.053，ICIR = -0.499，|IC|>0.02占比 = 86%
- G1年化+14.4%（Sharpe=0.48），G5年化+4.2%（Sharpe=0.13），单调性良好
- LS年化-8.46%，LS Sharpe=-0.79（IC 与 LS 方向一致）

**数据**：`close_adj`（由 `USE_ADJ_PRICE` 控制）

**参数**：`MIN_ROLLING_VALID_DAYS = 10`

---

### 8. overnight_ret — 隔夜收益率

**计算公式**：

$$\text{OVN}_{i,d} = \frac{\text{Open}_{i,d}}{\text{Close}_{i,d-1}} - 1$$

月内所有交易日隔夜收益率的均值。

**因子方向**：理论正向（高隔夜收益反映信息不对称溢价），但 A 股实证接近无效

**数据**：`open_adj`（或 `open`）、`close_adj`（或 `close`），由 `USE_ADJ_PRICE` 控制

**回测结果（2004-2026）**：
- IC均值 = +0.010，ICIR = +0.13，LS年化仅+0.8%，**基本无效**

**A股 T+1 制度影响**：当日买入无法当日卖出，使得隔夜信息无法被当日套利消化，
信号迅速被交易摩擦吸收。这是 A 股制度性特征导致的因子失效，非计算问题。

---

### 9. volatility_30 — 短期波动率

**计算公式**：

$$\text{VOL}_{i,t} = \text{std}(r_{i,d})_{d \in [t-29, t]}$$

过去 30 个交易日日度收益率的标准差，月末取值。

**因子方向**：负向（低波动率 → 高未来收益）—— 低波动异象（Low Volatility Anomaly）

**数据**：`close_adj`（由 `USE_ADJ_PRICE` 控制）

**参数**：`VOLATILITY_WINDOW = 30`

**回测结果（2004-2026）**：
- IC均值 = -0.064，ICIR = -0.37，|IC|>0.02占比 = 89%
- G1年化+14.2%（Sharpe=0.52），G5年化+3.7%

**注**：G2/G3 年化收益略高于 G1（15.8%/16.5% > 14.2%），
极低波动股票（可能为长期横盘/停牌边缘股）表现略弱，甜蜜点在 G2-G3。

---

## 基本面因子（`factors/fundamental.py`）

---

### 10. pb — 市净率

**数据来源**：Database `pb.csv`（2025-06 以后有完整数据；历史数据待补全）

**中性化**：市值+行业双重中性化（与 `net_profit_yoy` 一致）

**当前状态**：数据不完整，回测区间受限，待 pb 历史数据补全后激活

---

### 11. pe_ttm — 市盈率 TTM

**数据来源**：Database `pe_ttm.csv`（2023-01-01 至今）

**处理**：负 PE（亏损公司）置 NaN，PE 对亏损股无意义

**当前状态**：仅覆盖 2023 年以后，回测周期过短

---

### 12. dividend_yield — 股息率 TTM

**数据来源**：Database `dividend_ratio.csv`（2025-06 以后有完整数据）

**当前状态**：历史数据缺口待补全

---

### 13. size — 市值因子

**计算公式**：$\text{SIZE} = \ln(\text{close} \times \text{float\_shares})$

不做中性化（市值因子本身是中性化的控制变量，不宜再对自身中性化）。

**数据**：`close`（不复权）、`float_shares`

---

### 14. net_profit_yoy — 净利润同比增速

**计算公式**：$\text{YoY}_t = \frac{NP_t}{NP_{t-4}} - 1$（季度同比）

**滞后处理**：滞后 3 个季度（财报披露延迟，默认值 `NP_YOY_LAG_QUARTERS = 3`）

**数据来源**：仅存档 `_archive/raw_data/归属母公司净利润.csv`（2014-2021）

**当前状态**：唯一仍依赖存档的因子，回测区间受限于存档时间范围

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

| 因子 | IC均值 | ICIR | LS年化 | LS最大回撤 | 单调性 | 有效性 |
|------|--------|------|--------|-----------|--------|--------|
| turnover_20 | -0.096 | **-0.72** | -20.0% | -99.3% | ★★★★★ | **强有效** |
| cs_spread | -0.057 | -0.46 | -8.5% | -87.9% | ★★★ | 有效 |
| reversal_20 | -0.066 | -0.44 | -15.8% | -97.9% | ★★★★ | 有效 |
| volatility_30 | -0.064 | -0.37 | -8.1% | -86.0% | ★★★★ | 有效 |
| amihud | +0.049 | +0.31 | +16.6% | **-24.9%** | ★★★★★ | **有效且稳健** |
| amihud_zero_adj | +0.044 | +0.27 | +14.0% | -30.4% | ★★★★ | 有效 |
| roll_spread | — | — | — | — | — | **待重新验证**（Cov≥0改NaN后） |
| overnight_ret | +0.010 | +0.13 | +0.8% | -43.8% | ★ | 无效 |
| momentum_12_1 | -0.014 | -0.08 | -2.9% | -65.8% | ★ | **无效**（A股动量反转） |

> roll_spread 的上一版本结果因 Cov≥0 置 0 导致 G5 仅有 13 个有效月份，数据失真，
> 不具参考价值，修改后需重新运行。

---

## 参考文献

- Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*.
- Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance*.
- Fong, K. Y. L., Holden, C. W., & Trzcinka, C. A. (2017). What are the best liquidity proxies for global research? *Review of Finance*.
- Goyenko, R. Y., Holden, C. W., & Trzcinka, C. A. (2009). Do liquidity measures measure liquidity? *Journal of Financial Economics*.
- Pastor, L., & Stambaugh, R. F. (2003). Liquidity risk and expected stock returns. *Journal of Political Economy*.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance*.
