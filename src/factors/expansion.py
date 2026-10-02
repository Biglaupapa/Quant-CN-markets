# =============================================================================
# factors/expansion.py
# 特征扩充因子库（2026-08-18 新建）
#
# 目的：为机器学习面板准备足够宽的特征集。现有 37 个因子对 GBRT/RF 偏窄——
# 树模型的价值在于特征间的非线性交互，特征数不足则交互空间撑不起来。
#
# 入选标准（用户 2026-08-18 定）：
#   1. 德国项目（Gu-Kelly-Xiu 德股复现）有清晰构建方法
#   2. 且我们用 Choice 现有字段就能构建
#   3. **不做估值反推**——不从 MV/PE 倒推净利润这类财务水平量。
#      财报数据优先，没有就暂时不做，避免损害数据质量。
#   4. 不做行业中性化的第二套特征（相关性过高，对树模型是冗余噪音）
#
# 因此本模块只含**纯量价 / 成交**衍生的特征，不含任何基本面增长类。
# 缺口清单（约 45 个需财报明细的特征）见 FACTORS.md。
#
# 数据来源标注：
#   [gkx]  对应 Gu-Kelly-Xiu (2020) / JKP 特征库的同名变量
#   [af]   移植自 AF-pricing/Coding/factorcal.py
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional
import logging

from src.data.loader import load_data, to_monthly
from src.data.universe import apply_universe, build_investable_mask
from src.factors.base import preprocess
from src.config.settings import USE_ADJ_PRICE, MIN_ROLLING_VALID_DAYS

log = logging.getLogger(__name__)

_CLOSE = "close_adj" if USE_ADJ_PRICE else "close"


# =============================================================================
# 内部工具
# =============================================================================

def _monthly_close(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """月度后复权收盘价（已过滤股票池）。动量族共用。"""
    close = load_data([_CLOSE], start=start, end=end)[_CLOSE]
    mask = build_investable_mask(start=start, end=end, freq="D")
    return to_monthly(apply_universe(close, mask), method="last")


def _daily_ret(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """日度收益率（已过滤股票池）。波动族共用。"""
    close = load_data([_CLOSE], start=start, end=end)[_CLOSE]
    mask = build_investable_mask(start=start, end=end, freq="D")
    return apply_universe(close, mask).pct_change()


def _mom(start: Optional[str], end: Optional[str],
         long_m: int, skip_m: int) -> pd.DataFrame:
    """通用动量：P_{t-skip} / P_{t-long} − 1。"""
    c = _monthly_close(start, end)
    return preprocess(c.shift(skip_m) / c.shift(long_m) - 1)


# =============================================================================
# 一、动量 / 反转族（对应 GKX 的 ret_*_* 系列）
#
# GKX/JKP 用一整组不同窗口的 ret_j_k 而非单一动量，因为不同持有期的
# 动量-反转结构在截面上并不相同：短端（1月）是反转，中端（6~12月）是动量，
# 长端（36月以上）又转为反转。树模型能自行学出这个期限结构，
# 前提是我们把各个窗口都喂给它。
# =============================================================================

def calc_ret_2_1(start=None, end=None) -> pd.DataFrame:
    """[gkx] ret_2_1：2 个月前至 1 个月前的收益（短端反转）。"""
    return _mom(start, end, 2, 1)


def calc_ret_3_1(start=None, end=None) -> pd.DataFrame:
    """[gkx] ret_3_1：3-1 月动量。"""
    return _mom(start, end, 3, 1)


def calc_ret_6_1(start=None, end=None) -> pd.DataFrame:
    """[gkx] ret_6_1：6-1 月动量。GKX 变量重要性里排名很高的一个。"""
    return _mom(start, end, 6, 1)


def calc_ret_6_0(start=None, end=None) -> pd.DataFrame:
    """[gkx] ret_6_0：含最近 1 月的 6 月收益（不跳月，与 ret_6_1 对照）。"""
    return _mom(start, end, 6, 0)


def calc_ret_9_1(start=None, end=None) -> pd.DataFrame:
    """[gkx] ret_9_1：9-1 月动量。"""
    return _mom(start, end, 9, 1)


def calc_ret_12_7(start=None, end=None) -> pd.DataFrame:
    """[gkx] ret_12_7：12 至 7 月前的收益（中段动量，剔除近半年）。"""
    return _mom(start, end, 12, 7)


def calc_ret_36_13(start=None, end=None) -> pd.DataFrame:
    """[gkx] 长期反转：36 至 13 月前的累计收益（De Bondt-Thaler 1985）。"""
    return _mom(start, end, 36, 13)


def calc_seasonality(start=None, end=None) -> pd.DataFrame:
    """
    [gkx] seas_6_10na：季节性动量（Heston-Sadka 2008）。

    取过去第 6~10 年中**同一日历月**的平均收益。A 股有明显的日历效应
    （春节前后、年报季），该特征捕捉「某只股票在某个月份系统性表现好」。

    实现：月度收益按 lag 72/84/.../120 个月（6~10 年）取均值。
    """
    c = _monthly_close(start, end)
    ret_m = c.pct_change()
    lags = [12 * y for y in range(6, 11)]          # 72, 84, 96, 108, 120
    stack = [ret_m.shift(l) for l in lags]
    seas = sum(stack) / len(stack)
    return preprocess(seas)


def calc_prc_high_252(start=None, end=None) -> pd.DataFrame:
    """
    [gkx] prc_highprc_252d：52 周高点接近度（George-Hwang 2004）。

        当前价 / 过去 252 个交易日最高价

    越接近 1 说明越靠近一年高点。文献上该值高 → 后续收益高
    （投资者对创新高存在锚定效应，反应不足）。
    """
    close = load_data([_CLOSE], start=start, end=end)[_CLOSE]
    mask = build_investable_mask(start=start, end=end, freq="D")
    close = apply_universe(close, mask)

    high_252 = close.rolling(252, min_periods=120).max()
    ratio = close / high_252.where(high_252 > 0)
    return preprocess(to_monthly(ratio, method="last"))


# =============================================================================
# 二、波动 / 极值族
# =============================================================================

def _rvol(start, end, window: int) -> pd.DataFrame:
    """已实现波动率：日收益滚动标准差，年化。"""
    ret = _daily_ret(start, end)
    vol = ret.rolling(window, min_periods=max(MIN_ROLLING_VALID_DAYS,
                                              window // 3)).std() * np.sqrt(252)
    return preprocess(to_monthly(vol, method="last"))


def calc_rvol_21(start=None, end=None) -> pd.DataFrame:
    """[gkx] rvol_21d：21 日已实现波动率（现有 volatility_30 的短窗口版）。"""
    return _rvol(start, end, 21)


def calc_rvol_252(start=None, end=None) -> pd.DataFrame:
    """[gkx] rvol_252d：252 日已实现波动率（长窗口，更稳定的风险度量）。"""
    return _rvol(start, end, 252)


def calc_rmax1_21(start=None, end=None) -> pd.DataFrame:
    """
    [gkx] rmax1_21d：MAX effect（Bali-Cakici-Whitelaw 2011）。

    过去 21 个交易日的**单日最大收益**。彩票型偏好——散户追逐有过
    极端上涨的股票，推高其价格，导致后续收益偏低（预期负向）。
    A 股散户占比高，该效应文献上比美股更强。
    """
    ret = _daily_ret(start, end)
    rmax = ret.rolling(21, min_periods=MIN_ROLLING_VALID_DAYS).max()
    return preprocess(to_monthly(rmax, method="last"))


def calc_rmax5_21(start=None, end=None) -> pd.DataFrame:
    """[gkx] rmax5_21d：过去 21 日**前 5 大**单日收益的均值（MAX 的稳健版）。"""
    ret = _daily_ret(start, end)
    rmax5 = ret.rolling(21, min_periods=MIN_ROLLING_VALID_DAYS).apply(
        lambda x: np.mean(np.sort(x[~np.isnan(x)])[-5:])
        if (~np.isnan(x)).sum() >= 5 else np.nan,
        raw=True,
    )
    return preprocess(to_monthly(rmax5, method="last"))


def calc_skew_21(start=None, end=None) -> pd.DataFrame:
    """
    特质偏度（21 日日收益偏度）。

    负偏度 = 崩盘风险高。文献上高偏度（右偏，彩票型）→ 后续收益低，
    与 MAX effect 同源但度量不同。
    """
    ret = _daily_ret(start, end)
    sk = ret.rolling(21, min_periods=MIN_ROLLING_VALID_DAYS).skew()
    return preprocess(to_monthly(sk, method="last"))


# =============================================================================
# 三、流动性 / 微观结构族
# =============================================================================

def calc_zero_trades_252(start=None, end=None) -> pd.DataFrame:
    """
    [gkx] zero_trades_252d —— Liu (2006) 的标准化换手率调整零成交天数（LM）。

        LM_x = [ #零成交天数 + (1 / 区间累计换手率) / Deflator ] × (21x / NoTD)

    直觉：主排序是「一年里有多少天想交易也交易不了」；**第二项是换手率的
    倒数，唯一作用是在零成交天数相同时打破平局**——两只都是 0 天零成交的
    股票，换手率低的那只流动性更差。Deflator 取 11,000 使该项落在 (0,1)，
    保证它永远不会翻转主排序。末项 21x/NoTD 把不同长度的窗口标准化到可比。

    ── 为什么必须带打破平局项（2026-08-20 修正）────────────────────────────
    初版写成「21 日零成交占比」，**两处都错**：窗口该是 252 日（GKX/JKP 的
    变量就叫 `zero_trades_252d`），且漏了 Liu 的换手率项。后果是截面上几乎
    全是平局，实测各截面：

        构建方式                 2012-12         2018-06         2024-12
        ① 21日占比（初版）    16个取值/96.1%   21个/96.4%      9个/99.7%
        ② 仅改252日窗口      124个/30.5%     202个/66.9%     15个/97.0%
        ③ Liu LM_252（现版）  2274/2278      3150/3152      5191/5192

    **只改窗口不够**——2016 年后监管严打随意停牌，停牌已极罕见，2024 年截面
    众数仍占 97%。换手率打破平局项是必需的。

    初版的退化造成了两个可见后果：`qcut` 分位边界重复导致五分组回测
    `division by zero`（当时误判为「数据的真实性质」而关闭了该因子）；
    以及树模型在薄样本窗口上拿它切出一小撮停牌股，产生 45.68% 的虚假重要性。

    ── 计算顺序的坑 ────────────────────────────────────────────────────────
    ⚠️ **必须在应用股票池掩码之前统计**。`build_investable_mask` 要求
    `trade_status == 1`（当日有成交），先套掩码会把所有停牌日抹成 NaN，
    于是投资域内零成交天数恒为 0、截面标准差为 0、标准化除零 → 整张表 NaN。
    正确做法是用原始序列统计，最后只在**输出的月度截面**上套掩码。
    """
    W = 252
    DEFLATOR = 11000.0          # Liu (2006) 月频口径取值，使平局项 ∈ (0,1)
    MIN_D = 120                 # 至少半年数据才给值

    d = load_data(["trade_status", "turn"], start=start, end=end)
    st, turn = d.get("trade_status"), d.get("turn")
    if st is None or turn is None:
        log.warning("[zero_trades_252] 缺少 trade_status 或 turn 数据")
        return pd.DataFrame()

    # trade_status: 1=有成交, 0=停牌。存档格式为字符串"交易"，此处兼容
    if st.dtypes.astype(str).eq("object").any():
        st = st.replace({"交易": 1, "停牌": 0})
    st = st.apply(pd.to_numeric, errors="coerce")

    not_traded = (st == 0).astype(float).where(st.notna())
    zeros = not_traded.rolling(W, min_periods=MIN_D).sum()
    n_obs = not_traded.rolling(W, min_periods=MIN_D).count()
    tsum  = turn.rolling(W, min_periods=MIN_D).sum()      # 区间累计换手率(%)

    with np.errstate(divide="ignore", invalid="ignore"):
        tiebreak = (1.0 / tsum.where(tsum > 0)) / DEFLATOR
        lm = (zeros + tiebreak) * (W / n_obs.where(n_obs > 0))

    lm_m = to_monthly(lm, method="last")
    mask_m = to_monthly(build_investable_mask(start=start, end=end, freq="D"),
                        method="last")
    lm_m = lm_m.where(mask_m.reindex_like(lm_m).fillna(False).astype(bool))
    return preprocess(lm_m)


def calc_dolvol_126(start=None, end=None) -> pd.DataFrame:
    """
    [gkx] dolvol_126d：过去 126 个交易日日均成交额的对数。

    与 Amihud 的区别：dolvol 是**交易活跃度水平**（越大越流动），
    Amihud 是**价格冲击**（越大越不流动）。两者相关但非同一维度。
    取 log 因为成交额是重尾分布。
    """
    amt = load_data(["amt"], start=start, end=end)["amt"]
    mask = build_investable_mask(start=start, end=end, freq="D")
    amt = apply_universe(amt, mask)

    dv = amt.rolling(126, min_periods=60).mean()
    return preprocess(to_monthly(np.log(dv.where(dv > 0)), method="last"))


def calc_turn_std_21(start=None, end=None) -> pd.DataFrame:
    """
    换手率波动（21 日换手率标准差 / 均值，即变异系数）。

    Chordia-Subrahmanyam-Anshuman (2001)：换手率的**波动**本身被定价，
    且与换手率**水平**方向相反。用变异系数而非原始标准差以剥离水平效应。
    """
    turn = load_data(["turn"], start=start, end=end)["turn"]
    mask = build_investable_mask(start=start, end=end, freq="D")
    turn = apply_universe(turn, mask)

    m = turn.rolling(21, min_periods=MIN_ROLLING_VALID_DAYS).mean()
    s = turn.rolling(21, min_periods=MIN_ROLLING_VALID_DAYS).std()
    cv = s / m.where(m > 0)
    return preprocess(to_monthly(cv, method="last"))


def calc_close_vwap_dev(start=None, end=None) -> pd.DataFrame:
    """
    ★ 收盘价相对 VWAP 的偏离度（月均）：close / vwap − 1

    **这是我们相对 GKX / JKP 的独有特征**——美国的学术数据库（CRSP）不提供
    日内 VWAP，因此该类特征在英文文献里几乎见不到。Choice 提供
    `vwap.csv`（= AMOUNT / VOLUME，全精度），2005 起全覆盖。

    经济含义：收盘价高于当日成交均价 → 尾盘买压强（资金在收盘前抢筹）；
    低于均价 → 尾盘抛压。是**日内资金流向**的直接代理，
    比隔夜收益（overnight_ret）更贴近真实交易行为。

    用后复权口径：close_adj 与 vwap_adj 同口径，比值消去复权因子，与不复权比值相同。
    2026-10-02 由不复权 close / vwap（均为派生，第③级）改为 close_adj / vwap_adj
    （Choice CLOSE / AVERAGE 直接字段，第①级），见 Database/docs/【登记】指标权威来源.md。
    """
    d = load_data(["close_adj", "vwap_adj"], start=start, end=end)
    close, vwap = d.get("close_adj"), d.get("vwap_adj")
    if close is None or vwap is None:
        log.warning("[close_vwap_dev] 缺少 close_adj 或 vwap_adj 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    close = apply_universe(close, mask)
    vwap = apply_universe(vwap, mask)

    dev = close / vwap.where(vwap > 0) - 1
    return preprocess(to_monthly(dev, method="mean"))


def calc_amihud_vwap(start=None, end=None) -> pd.DataFrame:
    """
    ★ VWAP 口径的 Amihud 非流动性。

    标准 Amihud 用收盘价算日收益，受尾盘操纵和收盘集合竞价影响。
    改用 VWAP 计算日收益后，价格冲击的度量更贴近**平均成交成本**，
    对小盘股尤其稳健。同样是 Choice 数据的独有优势。
    """
    d = load_data(["vwap_adj", "amt"], start=start, end=end)
    vwap, amt = d.get("vwap_adj"), d.get("amt")
    if vwap is None or amt is None:
        log.warning("[amihud_vwap] 缺少 vwap_adj 或 amt 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    vwap = apply_universe(vwap, mask)
    amt = apply_universe(amt, mask)

    ret = vwap.pct_change()
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = (ret.abs() / amt.where(amt > 0)) * 1e5
    m = to_monthly(illiq, method="mean")
    return preprocess(np.log(m.where(m > 0)))


def calc_high_low_range(start=None, end=None) -> pd.DataFrame:
    """
    日内振幅（月均）：(high − low) / close

    最朴素的日内波动度量，与 cs_spread（Corwin-Schultz）同源但不做
    价差推断，保留原始振幅信息。A 股涨跌停制度下振幅有上限，
    该特征也隐含了「是否触及涨跌停」的信息。
    """
    d = load_data(["high_adj", "low_adj", _CLOSE], start=start, end=end)
    high, low, close = d.get("high_adj"), d.get("low_adj"), d.get(_CLOSE)
    if high is None or low is None or close is None:
        log.warning("[high_low_range] 缺少 high_adj / low_adj / close 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    high, low, close = (apply_universe(x, mask) for x in (high, low, close))

    rng = (high - low) / close.where(close > 0)
    return preprocess(to_monthly(rng, method="mean"))


# =============================================================================
# 四、规模 / 股本族
#
# 注：这里**只用原始下载字段**，不做估值反推。
# `chcsho_12m`（总股本变动）需由 MV/close 反推总股本，属反推法，故不做。
# `float_shares` 是 Choice 原始字段（LIQSHARE），其变化率合法。
# =============================================================================

def calc_free_float_ratio(start=None, end=None) -> pd.DataFrame:
    """
    ★ 流通占比：流通股本 / 总股本（因子名沿用 free_float_ratio，实为流通股本口径）

    **A 股特色特征**，在美股几乎无意义（美股基本全流通），
    但 A 股存在大量限售股、国有股，流通比例差异极大。
    比例低 → 筹码集中、易被操纵、流动性差。
    2026-10-02 由 neg_market_value / market_value 改为 float_shares / total_shares：
    两个都是 Choice 直接字段（LIQSHARE / TOTALSHARE），同源同生效日；流通市值换成财汇后，
    旧式会变成分子财汇、分母 Choice，股本生效日错位。见 Database/docs/【登记】指标权威来源.md。
    """
    d = load_data(["float_shares", "total_shares"], start=start, end=end)
    neg, tot = d.get("float_shares"), d.get("total_shares")
    if neg is None or tot is None:
        log.warning("[free_float_ratio] 缺少 float_shares 或 total_shares")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    neg, tot = apply_universe(neg, mask), apply_universe(tot, mask)

    ratio = neg / tot.where(tot > 0)
    return preprocess(to_monthly(ratio, method="last"))


def calc_float_shares_chg(start=None, end=None) -> pd.DataFrame:
    """
    ★ 流通股本 12 月变化率：float_shares_t / float_shares_{t-12} − 1

    捕捉**限售股解禁**压力。A 股的解禁是可预期的供给冲击，
    解禁量大 → 抛压大 → 后续收益低（预期负向）。
    `float_shares` 是 Choice 原始字段 LIQSHARE，非反推。
    """
    ffs = load_data(["float_shares"], start=start, end=end).get("float_shares")
    if ffs is None:
        log.warning("[float_shares_chg] 缺少 float_shares 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    ffs_m = to_monthly(apply_universe(ffs, mask), method="last")

    chg = ffs_m / ffs_m.shift(12).where(ffs_m.shift(12) > 0) - 1
    return preprocess(chg)


def calc_age(start=None, end=None) -> pd.DataFrame:
    """
    [gkx] age：上市年限（log(上市交易日数)）。

    次新股有明显的估值溢价与高波动。虽然股票池已剔除上市 <60 日的，
    但 1~3 年的次新股仍与老股行为不同，该特征让模型能区分。
    数据直接来自 Choice 的 `listed_days`，无推算。
    """
    ld = load_data(["listing_days"], start=start, end=end).get("listing_days")
    if ld is None:
        log.warning("[age] 缺少 listing_days 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    ld = apply_universe(ld, mask)
    return preprocess(to_monthly(np.log(ld.where(ld > 0)), method="last"))


# =============================================================================
# 五、估值比率的时序变换
#
# 只对**已有的原始估值比率**做变换（取倒数、算变化率），
# 不反推任何财务水平量。
# =============================================================================

def _inv_ratio(field: str, start, end, name: str) -> pd.DataFrame:
    """估值比率取倒数（收益率口径）。负值置 NaN——亏损公司的 E/P 不可比。"""
    x = load_data([field], start=start, end=end).get(field)
    if x is None:
        log.warning(f"[{name}] 缺少 {field} 数据")
        return pd.DataFrame()
    mask = build_investable_mask(start=start, end=end, freq="D")
    x = apply_universe(x, mask)
    inv = 1.0 / x.where(x > 0)
    return preprocess(to_monthly(inv, method="last"))


def calc_ep(start=None, end=None) -> pd.DataFrame:
    """
    [gkx] ni_me：盈利收益率 E/P = 1 / PE_TTM。

    与 pe_ttm 因子的区别：取倒数后**单调性更好**。PE 在零附近发散
    （微利公司 PE 极大），E/P 则连续；负值（亏损）置 NaN 而非留在
    分布另一端，避免「亏损公司看起来最便宜」的排序错误。
    """
    return _inv_ratio("pe_ttm", start, end, "ep")


def calc_ep_lsy(start=None, end=None) -> pd.DataFrame:
    """
    LSY（2019）口径的盈利收益率：EP = 1 / 市盈率（TTM、扣除非经常性损益）。

    原文："Earnings equals the most recently reported annualized net profit excluding
    non-recurrent gains/losses"；"We keep negative EP stocks ... and categorize them as
    growth stocks"——**负 EP 保留**（与 `ep` 把亏损置 NaN 不同）。
    数据：财汇 tq_sk_finindic.PETTMNPAAEI。原文「annualized」有歧义：最新报告期年化（PEMRQNPAAEI）
    与官方 VMG 相关 0.921，TTM 扣非 0.961（20 组对比，2026-10-02），故取 TTM（Louis 同意）。
    与 `ep`（Choice PETTM，TTM、含非经常性损益）是并存的相似指标，见 Database/docs/【登记】指标权威来源.md。
    """
    pe = load_data(["pe_ttm_deducted"], start=start, end=end).get("pe_ttm_deducted")
    if pe is None:
        log.warning("[ep_lsy] 缺少 pe_ttm_deducted 数据")
        return pd.DataFrame()
    mask = build_investable_mask(start=start, end=end, freq="D")
    pe = apply_universe(pe, mask)
    return preprocess(to_monthly(1.0 / pe.where(pe != 0), method="last"))


def calc_sp(start=None, end=None) -> pd.DataFrame:
    """[gkx] sale_me：营收收益率 S/P = 1 / PS_TTM。ps_ttm 覆盖 100%、无负值。"""
    return _inv_ratio("ps_ttm", start, end, "sp")


def calc_pb_chg_12(start=None, end=None) -> pd.DataFrame:
    """
    PB 的 12 月变化率：估值扩张 / 收缩速度。

    水平（pb）与变化（pb_chg）是两个维度：便宜的股票不一定在变便宜。
    这是对**已有原始字段**的时序变换，不涉及反推。
    """
    pb = load_data(["pb"], start=start, end=end).get("pb")
    if pb is None:
        log.warning("[pb_chg_12] 缺少 pb 数据")
        return pd.DataFrame()
    mask = build_investable_mask(start=start, end=end, freq="D")
    pb_m = to_monthly(apply_universe(pb, mask), method="last")
    prev = pb_m.shift(12)
    return preprocess(pb_m / prev.where(prev > 0) - 1)
