# =============================================================================
# factors/microstructure.py
# 微观结构因子库
#
# 包含所有价格/成交量衍生的市场微观结构信号，分两组：
#
# ── 活跃因子（可直接计算）─────────────────────────────────────────────────
#   reversal_20              短期反转（20日累计收益）
#   momentum_12_1            中期动量（12-1月，跳过最近1月）
#   turnover_20              换手率（20日均值，无中性化）
#   turnover_20_neutral      换手率（20日均值，流通市值中性化）
#   amihud                   Amihud 非流动性（3月滚动，成交额口径，无中性化）
#   amihud_neutral           Amihud 非流动性（3月滚动，流通市值中性化）
#   amihud_zero_adj          Amihud 零交易日调整版（log+NT修正，无中性化）
#   amihud_zero_adj_neutral  Amihud 零交易日调整版（log+NT修正，流通市值中性化）
#   cs_spread                Corwin-Schultz 高低价价差
#   roll_spread              Roll 价差（相邻收益率协方差）
#   overnight_ret            隔夜收益率（月均）
#   volatility_30            短期波动率（30日，月末取值）
#
# ── 风险 / 流动性风险因子（2026-08-18 全部激活）──────────────────────────
#   ivol             特质波动率（FF3 残差年化标准差）
#   ff3_betas        FF3 市场 Beta（= β_mkt）
#   beta_smb         FF3 规模载荷 ★
#   beta_hml         FF3 价值载荷 ★
#   capm_beta        CAPM 市场 Beta（240日滚动）★
#   ps_gamma         Pastor-Stambaugh Gamma ★
#   ps_liq_beta      PS 流动性 Beta（36月滚动）★
#   ap_beta1~5       Acharya-Pedersen 五个流动性 Beta（36月滚动）★
#
#   ★ = 本轮激活。原「需补 marketrtn_daily.csv」是伪缺口：
#      ff3_daily.csv 的 MKT 是超额市场收益，r_m = MKT + Rf 即所需序列。
#      详见 FACTORS.md「2026-08-18 新增」一节。
#
# 数据来源标注：
#   [arch]   来自 _archive/raw_data/ 历史 CSV（2014-2020）
#   [db]     来自 Database/data/stock/A/（2021-至今）
#   [af]     移植自 AF-pricing/Coding/factorcal.py
#   [iref]   移植自 IREF-sentiment/code/Data Processing.py
#   [orig]   移植自原 因子框架.py
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional, Dict
import functools
import logging
import warnings
from sklearn.linear_model import LinearRegression

# 模块级 logger。原先各因子的 `log.warning(...)` 错误分支引用了一个从未定义的
# 名字，一旦真的缺数据会抛 NameError 而不是打印警告（2026-08-18 补）。
log = logging.getLogger(__name__)

from src.data.loader import load_data, load_data_hk, to_monthly
from src.data.universe import apply_universe, build_investable_mask, build_investable_mask_hk
from src.factors.base import preprocess, winsorize, standardize
from src.config.settings import (
    REVERSAL_WINDOW,
    MOMENTUM_LONG,
    MOMENTUM_SKIP,
    TURNOVER_WINDOW,
    AMIHUD_WINDOW_MONTHS,
    AMIHUD_LAG_MONTHS,
    ILLIQ_SCALE,
    VOLATILITY_WINDOW,
    MIN_ROLLING_VALID_DAYS,
    USE_ADJ_PRICE,
    # 待激活因子参数（暂不调用）
    CAPM_BETA_WINDOW,
    IVOL_WINDOW_MONTHS,
    PS_GAMMA_MIN_OBS,
    PS_BETA_WINDOW_MONTHS,
    AP_BETA_WINDOW_MONTHS,
    AP_C1, AP_C2, AP_C3,
)
from src.factors.fundamental import _get_log_mktcap

# 价格字段选择（由 settings.USE_ADJ_PRICE 控制）
# 修改 settings.py 中的 USE_ADJ_PRICE 即可全局切换，无需逐函数修改
_CLOSE = "close_adj" if USE_ADJ_PRICE else "close"
_OPEN  = "open_adj"  if USE_ADJ_PRICE else "open"


# =============================================================================
# 市场路由辅助函数
# =============================================================================

def _get_market_loaders(market: str):
    """
    根据市场代码返回对应的数据加载函数和股票池掩码构建函数。

    Parameters
    ----------
    market : str
        "A"  → A 股：load_data + build_investable_mask（存档 + Database 合并）
        "HK" → 港股：load_data_hk + build_investable_mask_hk（仅 Database）

    Returns
    -------
    (loader_fn, mask_fn)
        loader_fn(fields, start, end) → dict
        mask_fn(start, end, freq) → pd.DataFrame
    """
    if market == "HK":
        return load_data_hk, build_investable_mask_hk
    return load_data, build_investable_mask


# =============================================================================
# ── 活跃因子 ──────────────────────────────────────────────────────────────
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 短期反转因子（20日累计收益）
#    来源：[orig] 因子框架.py
# -----------------------------------------------------------------------------
def calc_reversal_20(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    短期反转因子：过去 REVERSAL_WINDOW 个交易日的累计对数收益率。

    计算逻辑：
        Ret_t = close_t / close_{t-N} - 1
    使用后复权收盘价，应用股票池掩码后在滚动窗口中要求至少
    MIN_ROLLING_VALID_DAYS 个有效交易日。

    输出：月末截面值（日度计算 → 月末采样）
    """
    data = load_data([_CLOSE], start=start, end=end)
    close = data[_CLOSE]

    mask = build_investable_mask(start=start, end=end, freq="D")
    close = apply_universe(close, mask)

    # 滚动窗口有效数据检查
    valid_count = close.rolling(window=REVERSAL_WINDOW).count()
    close_filtered = close.copy()
    close_filtered[valid_count < MIN_ROLLING_VALID_DAYS] = np.nan

    # 20日累计收益
    ret20 = close_filtered / close_filtered.shift(REVERSAL_WINDOW) - 1

    # 月末采样
    factor = to_monthly(ret20, method="last")
    return preprocess(factor)


# -----------------------------------------------------------------------------
# 2. 中期动量因子（12-1月，跳过最近1月）
#    来源：[af] factorcal.py FactorMOM
# -----------------------------------------------------------------------------
def calc_momentum_12_1(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    中期动量因子：12个月前至1个月前的累计收益，跳过最近1个月以避免短期反转。

    计算逻辑（月度收盘价）：
        MOM_t = P_{t-1} / P_{t-12} - 1

    输出：月度截面
    """
    data = load_data([_CLOSE], start=start, end=end)
    close = data[_CLOSE]

    mask = build_investable_mask(start=start, end=end, freq="D")
    close = apply_universe(close, mask)

    # 降至月度收盘价
    close_m = to_monthly(close, method="last")

    # 12-1 动量
    factor = close_m.shift(MOMENTUM_SKIP) / close_m.shift(MOMENTUM_LONG) - 1
    return preprocess(factor)


# -----------------------------------------------------------------------------
# 3. 换手率因子（20日均值）
#    来源：[orig] 因子框架.py
#
#    两个版本：
#      turnover_20         无中性化（原始信号）
#      turnover_20_neutral 流通市值中性化（neg_market_value，Datayes）
# -----------------------------------------------------------------------------

def _calc_turnover_20_raw(
    start: Optional[str],
    end: Optional[str],
    market: str = "A",
) -> tuple:
    """内部辅助：返回 (factor_m, mask)，供两个版本共用。"""
    _load, _mask_builder = _get_market_loaders(market)

    data = _load(["turn"], start=start, end=end)
    turn = data.get("turn")

    if turn is None:
        warnings.warn("[microstructure] calc_turnover_20: 换手率数据加载失败，跳过")
        return None, None

    mask = _mask_builder(start=start, end=end, freq="D")
    turn = apply_universe(turn, mask)

    valid_count = turn.rolling(window=TURNOVER_WINDOW).count()
    turn[valid_count < MIN_ROLLING_VALID_DAYS] = np.nan

    turn20   = turn.rolling(window=TURNOVER_WINDOW).mean()
    factor_m = to_monthly(turn20, method="last")
    return factor_m, mask


def calc_turnover_20(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
) -> pd.DataFrame:
    """
    换手率因子（无中性化）：过去 TURNOVER_WINDOW 个交易日的平均换手率，月末取值。
    数据：turn [db]（A股 / 港股均可用）
    market : "A"（默认，A股）或 "HK"（港股）
    """
    factor_m, _ = _calc_turnover_20_raw(start, end, market=market)
    if factor_m is None:
        return pd.DataFrame()
    return preprocess(factor_m)


def calc_turnover_20_neutral(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
) -> pd.DataFrame:
    """
    换手率因子（流通市值中性化）：剔除换手率与 log(流通市值) 的线性相关后取残差。
    中性化变量：neg_market_value（Datayes，2004+，仅 A 股可用）
    数据：turn [db] + neg_market_value [db]
    注意：港股（market="HK"）无 neg_market_value，自动退化为无中性化版本。
    """
    factor_m, mask = _calc_turnover_20_raw(start, end, market=market)
    if factor_m is None:
        return pd.DataFrame()

    log_mktcap = _get_log_mktcap(start, end, mask, use_field="neg_market_value")
    if log_mktcap is None:
        warnings.warn("[microstructure] calc_turnover_20_neutral: 无法获取流通市值，退化为无中性化版本")
        return preprocess(factor_m)

    return preprocess(factor_m, neutralize="size", log_mktcap=log_mktcap)


# -----------------------------------------------------------------------------
# 4. Amihud 非流动性因子（版本A：3月滚动，成交额口径）
#    来源：[af] factorcal.py FactorAmihud
#
#    两个版本：
#      amihud         无中性化（原始信号）
#      amihud_neutral 流通市值中性化（neg_market_value，Datayes）
# -----------------------------------------------------------------------------

def _calc_amihud_factor(
    start: Optional[str],
    end: Optional[str],
    market: str = "A",
) -> tuple:
    """内部辅助：返回 (factor, mask)，供两个版本共用。"""
    _load, _mask_builder = _get_market_loaders(market)

    data  = _load(["close_adj", "amt"], start=start, end=end)
    close = data.get("close_adj") if data.get("close_adj") is not None else data.get("close")
    amt   = data.get("amt")

    if close is None or amt is None:
        warnings.warn("[microstructure] calc_amihud: 缺少 close 或 amt 数据")
        return None, None

    mask  = _mask_builder(start=start, end=end, freq="D")
    close = apply_universe(close, mask)
    amt   = apply_universe(amt, mask)

    abs_ret      = (close / close.shift(1) - 1).abs()
    amt_millions = (amt / 1e6).replace(0, np.nan)
    daily_illiq  = abs_ret / amt_millions * ILLIQ_SCALE

    daily_illiq.index = pd.to_datetime(daily_illiq.index)

    frames = []
    dates  = []
    buffer = []

    for period_end, group in daily_illiq.resample("ME"):
        buffer.append(group)
        if len(buffer) > AMIHUD_WINDOW_MONTHS:
            buffer.pop(0)
        if len(buffer) < AMIHUD_WINDOW_MONTHS:
            continue

        window_data = pd.concat(buffer)
        valid_count = window_data.count()
        row = window_data.mean()
        row[valid_count < MIN_ROLLING_VALID_DAYS] = np.nan

        frames.append(row)
        dates.append(period_end)

    if not frames:
        return None, None

    factor = pd.DataFrame(frames, index=pd.DatetimeIndex(dates))
    factor = factor.shift(AMIHUD_LAG_MONTHS)
    return factor, mask


def calc_amihud(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
) -> pd.DataFrame:
    """
    Amihud (2002) 非流动性因子（无中性化）：
        ILLIQ_{i,t} = mean(|R_{i,d}| / Amount_{i,d}) × ILLIQ_SCALE
    其中 d 取过去 3 个月的日度数据，滞后 1 个月以避免前瞻偏差。

    数据：close_adj, amt [db]（A股 / 港股均可用）
    因子方向：ILLIQ 越大 = 流动性越差。
    market : "A"（默认，A股）或 "HK"（港股）
    """
    factor, _ = _calc_amihud_factor(start, end, market=market)
    if factor is None:
        return pd.DataFrame()
    return preprocess(factor)


def calc_amihud_neutral(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
) -> pd.DataFrame:
    """
    Amihud 非流动性因子（流通市值中性化）：剔除与 log(流通市值) 的线性相关后取残差。
    中性化变量：neg_market_value（Datayes，2004+，仅 A 股可用）
    数据：close_adj, amt [db] + neg_market_value [db]
    注意：港股（market="HK"）无 neg_market_value，自动退化为无中性化版本。
    """
    factor, mask = _calc_amihud_factor(start, end, market=market)
    if factor is None:
        return pd.DataFrame()

    log_mktcap = _get_log_mktcap(start, end, mask, use_field="neg_market_value")
    if log_mktcap is None:
        warnings.warn("[microstructure] calc_amihud_neutral: 无法获取流通市值，退化为无中性化版本")
        return preprocess(factor)

    return preprocess(factor, neutralize="size", log_mktcap=log_mktcap)


# -----------------------------------------------------------------------------
# 5. Amihud 零交易日调整版（版本C：log + NT 修正）
#    来源：[iref] IREF-sentiment/code/Data Processing.py
#
#    两个版本：
#      amihud_zero_adj         无中性化（原始信号）
#      amihud_zero_adj_neutral 流通市值中性化（neg_market_value，Datayes）
# -----------------------------------------------------------------------------

def _calc_amihud_zero_adj_factor(
    start: Optional[str],
    end: Optional[str],
) -> tuple:
    """内部辅助：返回 (factor, mask)，供两个版本共用。"""
    data = load_data([_CLOSE, "volume"], start=start, end=end)
    close  = data.get(_CLOSE)
    volume = data.get("volume")

    if close is None or volume is None:
        warnings.warn("[microstructure] calc_amihud_zero_adj: 缺少 close 或 volume 数据")
        return None, None

    mask = build_investable_mask(start=start, end=end, freq="D")
    close  = apply_universe(close, mask)
    volume = apply_universe(volume, mask)

    abs_ret     = (close / close.shift(1) - 1).abs()
    vol_clean   = volume.replace(0, np.nan)
    daily_illiq = abs_ret / vol_clean

    frames = []
    dates  = []

    for period_end, group in daily_illiq.resample("ME"):
        n_total = len(group)
        if n_total == 0:
            continue

        nt = (volume.reindex(group.index) == 0).sum() / n_total

        mean_illiq = group.mean()
        mean_illiq = mean_illiq.replace([np.inf, -np.inf], np.nan)

        log_illiq = np.log(mean_illiq.clip(lower=1e-15))
        adjusted  = log_illiq * (nt + 1)

        frames.append(adjusted)
        dates.append(period_end)

    if not frames:
        return None, None

    factor = pd.DataFrame(frames, index=pd.DatetimeIndex(dates))
    return factor, mask


def calc_amihud_zero_adj(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Amihud 零交易日调整版（无中性化）：
        ILLIQ_ZA = ln(mean(|R| / Volume)) × (NT + 1)
    其中 NT 为当月零交易日占比。

    数据：close, volume [db]
    """
    factor, _ = _calc_amihud_zero_adj_factor(start, end)
    if factor is None:
        return pd.DataFrame()
    return preprocess(factor)


def calc_amihud_zero_adj_neutral(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Amihud 零交易日调整版（流通市值中性化）：剔除与 log(流通市值) 的线性相关后取残差。
    中性化变量：neg_market_value（Datayes，2004+）
    数据：close, volume [db] + neg_market_value [db]
    """
    factor, mask = _calc_amihud_zero_adj_factor(start, end)
    if factor is None:
        return pd.DataFrame()

    log_mktcap = _get_log_mktcap(start, end, mask, use_field="neg_market_value")
    if log_mktcap is None:
        warnings.warn("[microstructure] calc_amihud_zero_adj_neutral: 无法获取流通市值，退化为无中性化版本")
        return preprocess(factor)

    return preprocess(factor, neutralize="size", log_mktcap=log_mktcap)


# -----------------------------------------------------------------------------
# 6. CS 价差（Corwin-Schultz 高低价价差）
#    来源：[af] factorcal.py FactorCsspread + [iref]
# -----------------------------------------------------------------------------
def calc_cs_spread(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
) -> pd.DataFrame:
    """
    Corwin-Schultz (2012) 高低价价差估计：
        1. beta  = (ln(H_t/L_t))² + (ln(H_{t+1}/L_{t+1}))²
        2. gamma = (ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1})))²
        3. alpha = (√(2β) - √β) / (3-2√2) - √(γ/(3-2√2))
        4. spread = 2(e^α-1)/(1+e^α)，负值截断为 0

    月度平均，要求至少 MIN_ROLLING_VALID_DAYS 个有效值。

    数据：high_adj, low_adj [db]（A股 / 港股均可用）
        使用后复权高低价，避免除权日前后跨日价格不连续导致
        max(H_t, H_{t+1}) / min(L_t, L_{t+1}) 失真。
    market : "A"（默认，A股）或 "HK"（港股）
    """
    _load, _mask_builder = _get_market_loaders(market)

    data = _load(["high_adj", "low_adj"], start=start, end=end)
    high = data.get("high_adj")
    low  = data.get("low_adj")

    if high is None or low is None:
        warnings.warn("[microstructure] calc_cs_spread: 缺少 high_adj/low_adj 数据")
        return pd.DataFrame()

    mask = _mask_builder(start=start, end=end, freq="D")
    high = apply_universe(high, mask)
    low  = apply_universe(low,  mask)

    # 防止 H/L 出现 0 或负数
    high = high.replace(0, np.nan)
    low  = low.replace(0, np.nan)

    h_l_ln   = (np.log(high / low)) ** 2
    h_l_ln_1 = h_l_ln.shift(-1)                             # 下一日

    # 跨日高低
    high_2d  = pd.concat([high, high.shift(-1)]).groupby(level=0).max()
    low_2d   = pd.concat([low,  low.shift(-1) ]).groupby(level=0).min()
    gamma    = (np.log(high_2d / low_2d)) ** 2

    beta  = h_l_ln + h_l_ln_1
    k     = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0)   # 负值截断

    # 月度均值
    frames = []
    dates  = []
    for period_end, group in spread.resample("ME"):
        valid = group.count()
        row   = group.mean()
        row[valid < MIN_ROLLING_VALID_DAYS] = np.nan
        frames.append(row)
        dates.append(period_end)

    if not frames:
        return pd.DataFrame()

    factor = pd.DataFrame(frames, index=pd.DatetimeIndex(dates))

    return preprocess(factor)


# -----------------------------------------------------------------------------
# 7. Roll 价差（相邻收益率协方差）
#    来源：[iref] IREF-sentiment/code/Data Processing.py
# -----------------------------------------------------------------------------
def calc_roll_spread(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Roll (1984) 价差：
        Roll_t = 2 × √|Cov(r_t, r_{t-1})|
    仅使用负协方差（正协方差时置 0，表示流动性充足）。

    月度计算，数据：close [arch]/[db]
    """
    data = load_data([_CLOSE], start=start, end=end)
    close = data.get(_CLOSE)

    if close is None:
        warnings.warn("[microstructure] calc_roll_spread: 缺少 close 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    close = apply_universe(close, mask)

    ret = close / close.shift(1) - 1

    frames = []
    dates  = []

    # ── 向量化说明 ────────────────────────────────────────────────
    # 原实现是「月 × 股票」双重循环（260 × 5855 ≈ 150 万次 np.cov）。
    #
    # 关键细节：原来是 `s = group[col].dropna()` 后再配 s[1:] 与 s[:-1]，
    # 即**按非空序列相邻配对**，而不是按日历相邻。若直接用 shift(1) 会
    # 把停牌日也算进去，结果不同。
    #
    # 恰好等价的向量化写法：`ffill().shift(1)` 在每个非空位置上给出的
    # 正是「上一个非空值」——与 dropna 后错位一位完全一致。
    #     Y     = [1, NaN, 3, 4]
    #     ffill = [1,  1,  3, 4]  → shift(1) = [NaN, 1, 1, 3]
    #     有效位 (0,2,3) 上的配对：(3,1)、(4,3)，与 dropna 后的 s 相同。
    #
    # 协方差沿用 np.cov 的 ddof=1，且两侧各自去均值（原实现即如此）。
    for period_end, group in ret.resample("ME"):
        if len(group) < MIN_ROLLING_VALID_DAYS:
            frames.append(pd.Series(np.nan, index=group.columns))
            dates.append(period_end)
            continue

        prev = group.ffill().shift(1)
        pair = group.notna() & prev.notna()          # 有效配对掩码
        a = group.where(pair)                        # r_t
        b = prev.where(pair)                         # r_{t-1}

        n_pair = pair.sum()                          # = len(s) - 1
        # 原判据 len(s) >= MIN_ROLLING_VALID_DAYS  ⇔  n_pair >= MIN - 1
        enough = n_pair >= (MIN_ROLLING_VALID_DAYS - 1)

        with np.errstate(invalid="ignore", divide="ignore"):
            cov = ((a - a.mean()) * (b - b.mean())).sum() / (n_pair - 1)

        # Cov < 0：bid-ask bounce 可检测，Roll spread 有意义，赋正值
        # Cov ≥ 0：Roll 公式产生虚数，无法估计 spread，置 NaN（不是 0）
        # 理由：Fong(2017) 和 Goyenko(2009) 均指出，正样本自相关
        #   对应真实 serial correlation 接近零的高流动性股票，
        #   Roll 模型对这类股票没有判别力。
        #   文献习惯赋 0（用于流动性水平比较），但在截面因子排序中，
        #   大量 0 值聚集会导致 pd.qcut 分组崩溃（57% 数据同值）。
        #   改为 NaN：明确区分"无法估计"与"流动性充分"，
        #   在 Cov<0 的子集（约 43% 股票-月份）内部排序，经济含义清晰。
        row = 2 * np.sqrt((-cov).where(cov < 0))
        row[~enough] = np.nan

        frames.append(row)
        dates.append(period_end)

    if not frames:
        return pd.DataFrame()

    factor = pd.DataFrame(frames, index=pd.DatetimeIndex(dates))

    return preprocess(factor)


# -----------------------------------------------------------------------------
# 8. 隔夜收益率（月均）
#    来源：[iref] IREF-sentiment/code/Data Processing.py
# -----------------------------------------------------------------------------
def calc_overnight_ret(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    隔夜收益率：当日开盘价 / 前日收盘价 - 1 的月度均值。
    反映信息不对称和隔夜风险。
    数据：open [db], close [arch]/[db]
    """
    data = load_data([_OPEN, _CLOSE], start=start, end=end)
    open_p  = data.get(_OPEN)
    close   = data.get(_CLOSE)

    if open_p is None or close is None:
        warnings.warn("[microstructure] calc_overnight_ret: 缺少 open 或 close 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    open_p = apply_universe(open_p, mask)
    close  = apply_universe(close,  mask)

    # 对齐列（open 来自 Database，close 可能来自存档）
    common_cols = open_p.columns.intersection(close.columns)
    overnight   = open_p[common_cols] / close[common_cols].shift(1) - 1

    factor = to_monthly(overnight, method="mean")
    return preprocess(factor)


# -----------------------------------------------------------------------------
# 9. 短期波动率（30日）
#    来源：[iref] IREF-sentiment/code/Data Processing.py
# -----------------------------------------------------------------------------
def calc_volatility_30(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    短期波动率：过去 VOLATILITY_WINDOW 个交易日日度收益率的标准差，月末取值。
    波动率越高，股票风险越大。

    数据：close [arch]/[db]
    """
    data = load_data([_CLOSE], start=start, end=end)
    close = data.get(_CLOSE)

    if close is None:
        warnings.warn("[microstructure] calc_volatility_30: 缺少 close 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    close = apply_universe(close, mask)

    ret     = close / close.shift(1) - 1
    vol_30  = ret.rolling(window=VOLATILITY_WINDOW).std()

    factor = to_monthly(vol_30, method="last")
    return preprocess(factor)


# =============================================================================
# ── 流动性风险因子（2026-08-18 激活）──────────────────────────────────────
#
# 这四个因子的代码骨架自建库起就在，一直卡在「缺 marketrtn_daily.csv」。
# 实际上该序列无需另行下载：`ff3_daily.csv` 的 MKT 列按定义就是
# 全市场可投资股票的 **VW 超额收益**（见 ff3_builder.py:289
# `return (vw_ret - rf_aligned).rename('MKT')`），因此
#
#       r_m = MKT + Rf
#
# 即为所需的日度市场收益率。`ap_betas` 另需的「日度流通市值」就是
# `neg_market_value.csv`。两项数据 2005-01-04 起全覆盖，故一并激活。
#
# 数学口径移植自 AF-pricing/Coding/factorcal.py（FactorGamma /
# FactorLiquidityBeta / FactorLiquidityBetaAP / FactorBeta），但原实现
# 是「月份 × 股票」双重 sklearn 循环，在 5861 只股票上不可行，
# 此处一律改为批量最小二乘（与 `_rolling_ff3_ols` 同一套解法）。
# =============================================================================

def _load_market_return_daily(start: Optional[str] = None,
                              end: Optional[str] = None) -> Optional[pd.Series]:
    """日度市场收益率 r_m = MKT + Rf（取自 ff3_daily.csv）。

    MKT 是 **超额** 市场收益（ff3_builder 里已减去 Rf），加回 Rf 才是
    PS Gamma / CAPM Beta 所需的原始市场收益。
    """
    from pathlib import Path

    ff3_path = Path("/Users/louis/MyProjects/Database/data/factors/ff3_daily.csv")
    if not ff3_path.exists():
        log.warning(f"[market_ret] FF3 日度文件不存在：{ff3_path}")
        return None

    ff3 = pd.read_csv(ff3_path, index_col=0, parse_dates=True)
    mkt = (ff3["MKT"] + ff3["Rf"]).rename("mkt_ret")
    if start:
        mkt = mkt.loc[start:]
    if end:
        mkt = mkt.loc[:end]
    return mkt.dropna()


def _batch_ols(X: np.ndarray, Y: np.ndarray, min_obs: int):
    """一次拟合 N 列因变量：对每列取自己的非空行解 (X'X)β = X'y。

    X : T×K（**须已含截距列**）   Y : T×N（可含 NaN）
    返回 (beta N×K, n_obs N, ok N)。与逐列 `LinearRegression().fit()`
    解同一组正规方程；用 pinv 而非 solve，对奇异矩阵优雅退化。
    """
    T, K = X.shape
    N = Y.shape[1]

    M     = ~np.isnan(Y)
    Mf    = M.astype(np.float64)
    Y0    = np.where(M, Y, 0.0)
    n_obs = M.sum(axis=0)

    XtX = np.empty((N, K, K))
    for i in range(K):
        for j in range(i, K):
            v = Mf.T @ (X[:, i] * X[:, j])
            XtX[:, i, j] = v
            XtX[:, j, i] = v
    Xty = (X.T @ Y0).T

    beta = np.full((N, K), np.nan)
    ok = n_obs >= min_obs
    if ok.any():
        beta[ok] = np.einsum('nij,nj->ni', np.linalg.pinv(XtX[ok]), Xty[ok])
    return beta, n_obs, ok


@functools.lru_cache(maxsize=4)
def _ps_gamma_raw(start: Optional[str] = None,
                  end: Optional[str] = None) -> pd.DataFrame:
    """PS Gamma 的原始月度面板（未经 preprocess，供 ps_liq_beta 复用）。

    逐月对每只股票做日内回归（Pastor-Stambaugh 2003 式 (1)）：

        r^e_{i,d+1} = θ + φ·r_{i,d} + γ·sign(r^e_{i,d})·v_{i,d} + ε

    其中 r^e = r_i - r_m（个股减市场），v 为成交额（百万元）。
    γ < 0 表示「成交量推动价格反转」即流动性差；γ 越负越不流动。
    月内最少 PS_GAMMA_MIN_OBS(=10) 个有效交易日。
    """
    data = load_data([_CLOSE, "amt"], start=start, end=end)
    close, amt = data.get(_CLOSE), data.get("amt")
    if close is None or amt is None:
        log.warning("[ps_gamma] 缺少 close_adj 或 amt 数据")
        return pd.DataFrame()

    mkt = _load_market_return_daily(start=start, end=end)
    if mkt is None:
        return pd.DataFrame()

    idx = close.index.intersection(amt.index).intersection(mkt.index)
    close, amt, mkt = close.loc[idx], amt.loc[idx], mkt.loc[idx]

    ret     = close.pct_change()
    excess  = ret.sub(mkt, axis=0)              # r^e = r_i - r_m
    amt_mn  = amt / 1e6                          # 元 → 百万元

    # 右侧全部取滞后一日：sign(r^e_{i,d})·v_{i,d} 与 r_{i,d} 对齐到 d+1
    x_ret  = ret.shift(1)
    x_sgnv = np.sign(excess.shift(1)) * amt_mn.shift(1)
    x_sgnv = x_sgnv.replace([np.inf, -np.inf], np.nan)

    cols = close.columns
    periods = excess.index.to_period("M")
    rows, dates = [], []

    for ym in periods.unique():
        m = periods == ym
        Y = excess.values[m]                        # T×N 因变量
        A = x_ret.values[m]                         # T×N 回归元1（逐股不同）
        B = x_sgnv.values[m]                        # T×N 回归元2（逐股不同）
        if Y.shape[0] < PS_GAMMA_MIN_OBS:
            continue

        # 三者任一缺失该观测即无效
        M = ~(np.isnan(Y) | np.isnan(A) | np.isnan(B))
        n_obs = M.sum(axis=0)
        A0, B0, Y0 = (np.where(M, v, 0.0) for v in (A, B, Y))
        Mf = M.astype(np.float64)

        # 逐股的 3×3 正规方程，一次性堆叠求解（回归元逐股不同，
        # 故不能走 _batch_ols 的「共享 X」路径，需手工组装各阶矩）
        XtX = np.empty((len(cols), 3, 3))
        XtX[:, 0, 0] = n_obs
        XtX[:, 0, 1] = XtX[:, 1, 0] = A0.sum(axis=0)
        XtX[:, 0, 2] = XtX[:, 2, 0] = B0.sum(axis=0)
        XtX[:, 1, 1] = (A0 * A0).sum(axis=0)
        XtX[:, 1, 2] = XtX[:, 2, 1] = (A0 * B0).sum(axis=0)
        XtX[:, 2, 2] = (B0 * B0).sum(axis=0)
        Xty = np.column_stack([(Mf * Y0).sum(axis=0),
                               (A0 * Y0).sum(axis=0),
                               (B0 * Y0).sum(axis=0)])

        gamma = np.full(len(cols), np.nan)
        ok = n_obs >= PS_GAMMA_MIN_OBS
        if ok.any():
            sol = np.einsum('nij,nj->ni', np.linalg.pinv(XtX[ok]), Xty[ok])
            gamma[ok] = sol[:, 2]                   # γ = 第 3 个系数

        rows.append(pd.Series(gamma, index=cols))
        # 用**日历月末**标签，与 loader.to_monthly 的 resample("ME") 一致；
        # 若用当月最后交易日，遇到月末落在周末的月份就对不齐（实测 235 个月
        # 只有 156 个能与 to_monthly 的索引相交）。
        dates.append(ym.to_timestamp("M"))

    if not dates:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates)).sort_index()


def _calc_ps_gamma(start: Optional[str] = None,
                   end: Optional[str] = None) -> pd.DataFrame:
    """
    Pastor-Stambaugh (2003) Gamma 流动性指标（月度）。

    γ 是「成交量对次日收益反转的推动力度」：越负 → 单位成交额引起的价格
    反转越大 → 流动性越差。因此 **γ 低 = 不流动 = 预期收益高**，
    与 amihud 方向相反（amihud 高 = 不流动）。

    需要数据：close_adj（日度）、amt（日度，元）、ff3_daily（推市场收益）
    来源：[af] factorcal.py FactorGamma
    """
    gamma = _ps_gamma_raw(start=start, end=end)
    if gamma.empty:
        return pd.DataFrame()
    return preprocess(gamma)


def _calc_ps_liq_beta(start: Optional[str] = None,
                      end: Optional[str] = None) -> pd.DataFrame:
    """
    Pastor-Stambaugh (2003) 流动性 Beta（36 月滚动）。

    Step1 市场流动性水平：γ_t 的**流通市值加权**截面均值
    Step2 规模调整的流动性变化：Δγ_t（PS 原文用 (m_t/m_1) 缩放，此处
          用市值加权已隐含规模权重，直接取一阶差分）
    Step3 流动性冲击 L_t：对 Δγ_t 做 AR(1) 回归取残差（不可预测部分）
    Step4 逐股 36 月滚动回归 r_i = α + b·r_m + c·L_t + ε，取 c 为因子

    c > 0 表示该股在市场流动性恶化时跌得更多（流动性风险暴露高），
    按 PS 原文应要求更高的预期收益。

    需要数据：ps_gamma、neg_market_value、月度收益率、ff3_daily
    来源：[af] factorcal.py FactorLiquidityBeta
    """
    gamma = _ps_gamma_raw(start=start, end=end)
    if gamma.empty:
        return pd.DataFrame()

    # ── Step1：市值加权的市场流动性水平 ────────────────────────────
    mv_d = load_data(["neg_market_value"], start=start, end=end).get("neg_market_value")
    if mv_d is None:
        log.warning("[ps_liq_beta] 缺少 neg_market_value 数据")
        return pd.DataFrame()
    mv = to_monthly(mv_d, method="last").reindex(index=gamma.index,
                                                 columns=gamma.columns)

    w = mv.where(gamma.notna())
    w = w.div(w.sum(axis=1), axis=0)
    gamma_m = (gamma * w).sum(axis=1, min_count=1)

    # ── Step2-3：一阶差分 → AR(1) 残差 = 流动性冲击 ────────────────
    dg = gamma_m.diff()
    ar = pd.DataFrame({"y": dg, "x": dg.shift(1)}).dropna()
    if len(ar) < 24:
        log.warning("[ps_liq_beta] 流动性序列过短，无法估计 AR(1)")
        return pd.DataFrame()
    b = np.polyfit(ar["x"].values, ar["y"].values, 1)
    liq_shock = (ar["y"] - (b[0] * ar["x"] + b[1])).rename("L")

    # ── Step4：36 月滚动回归 r_i = α + b·r_m + c·L ─────────────────
    from src.backtest.engine import calc_monthly_returns
    ret_m = calc_monthly_returns(start=start, end=end, market="A")

    mkt_d = _load_market_return_daily(start=start, end=end)
    mkt_m = (1 + mkt_d).groupby(mkt_d.index.to_period("M")).prod() - 1
    mkt_m.index = mkt_m.index.to_timestamp("M")

    idx = ret_m.index.intersection(liq_shock.index).intersection(mkt_m.index)
    ret_m, L, rm = ret_m.loc[idx], liq_shock.loc[idx], mkt_m.loc[idx]

    cols, rows, dates = ret_m.columns, [], []
    for i in range(PS_BETA_WINDOW_MONTHS - 1, len(idx)):
        sl = slice(i - PS_BETA_WINDOW_MONTHS + 1, i + 1)
        X = np.column_stack([np.ones(PS_BETA_WINDOW_MONTHS),
                             rm.values[sl], L.values[sl]])
        Y = ret_m.values[sl]
        beta, _, ok = _batch_ols(X, Y, min_obs=24)
        rows.append(pd.Series(np.where(ok, beta[:, 2], np.nan), index=cols))
        dates.append(idx[i])

    if not dates:
        return pd.DataFrame()
    return preprocess(pd.DataFrame(rows, index=pd.DatetimeIndex(dates)))


@functools.lru_cache(maxsize=4)
def _ap_betas_raw(start: Optional[str] = None,
                  end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Acharya-Pedersen (2005) 五个流动性 Beta 的原始月度面板。

    先把 Amihud 非流动性折算成「交易成本」c_i（AP 原文式 (12)）：

        c_i,t = min(AP_C1 + AP_C2 · ILLIQ_i,t · mv_ratio_t, AP_C3)

    其中 mv_ratio 用市场总市值相对基期的比值做通胀调整，AP_C3(=60%) 是
    成本上限，防止极端不流动股把协方差矩阵带跑。随后 36 月滚动计算：

        β1 = Cov(R_i, R_M) / Var(R_M − C_M)     传统市场 Beta
        β2 = Cov(c_i, c_M) / Var(R_M − C_M)     流动性共动
        β3 = Cov(R_i, c_M) / Var(R_M − C_M)     收益对市场流动性的暴露
        β4 = Cov(c_i, R_M) / Var(R_M − C_M)     流动性对市场收益的暴露
        β5 = β2 − β3 − β4                       净流动性 Beta

    AP 原文的定价含义：β2 越高、β3/β4 越负 → 流动性风险越大 → 要求补偿。
    来源：[af] factorcal.py FactorLiquidityBetaAP
    """
    data = load_data([_CLOSE, "amt", "neg_market_value"], start=start, end=end)
    close = data.get(_CLOSE)
    amt   = data.get("amt")
    mv    = data.get("neg_market_value")
    if close is None or amt is None or mv is None:
        log.warning("[ap_betas] 缺少 close_adj / amt / neg_market_value 数据")
        return {}

    # ── 日度 Amihud 非流动性 → 月度均值 ───────────────────────────
    #
    # ⚠️ 这里**不能**用全局 ILLIQ_SCALE(=1e5)。AP 的 c1/c2/c3(0.25/0.30/60)
    # 是一组有量纲的标定值，只在 ILLIQ 落在 O(0.01~1) 时才有意义。
    # 参照实现 factorcal.py 的 `amount` 单位是**千元**，而 Database 的
    # `amt.csv` 单位是**元**（indicators.csv: 成交额(元)），差 1000 倍。
    # 沿用 1e5 会让 ILLIQ 中位数只有 2.6e-5，c ≡ 0.25 几乎完全是常数
    # （实测截面标准差 0.002 vs 水平 0.25），β2~β4 退化成纯噪声。
    # 故此处用 1e5 × 1000 = 1e8，与参照实现的标定对齐。
    #
    # 不改全局 ILLIQ_SCALE 是因为 amihud 因子依赖它，而对 amihud 来说
    # 它只是个单调缩放（不影响 IC/分组），改了反而会让既有缓存失效。
    AP_ILLIQ_SCALE = ILLIQ_SCALE * 1000.0

    ret = close.pct_change()
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq_d = (ret.abs() / amt.where(amt > 0)) * AP_ILLIQ_SCALE
    illiq_m = to_monthly(illiq_d, method="mean")

    ret_m = to_monthly(close, method="last").pct_change()
    mv_m  = to_monthly(mv, method="last")

    # ── 通胀调整比值：市场总市值 / 基期总市值 ─────────────────────
    mv_tot = mv_m.sum(axis=1, min_count=1)
    mv_ratio = (mv_tot / mv_tot.iloc[0]).clip(upper=30.0)

    # ── 折算为交易成本 c（上限 AP_C3）────────────────────────────
    cost = (AP_C1 + AP_C2 * illiq_m.mul(mv_ratio, axis=0)).clip(upper=AP_C3)

    # ── 市场侧：市值加权的 R_M 与 c_M ────────────────────────────
    w = mv_m.where(ret_m.notna())
    w = w.div(w.sum(axis=1), axis=0)
    R_M = (ret_m * w).sum(axis=1, min_count=1)
    wc = mv_m.where(cost.notna())
    wc = wc.div(wc.sum(axis=1), axis=0)
    c_M = (cost * wc).sum(axis=1, min_count=1)

    idx = ret_m.index.intersection(cost.index).intersection(R_M.dropna().index)
    ret_m, cost = ret_m.loc[idx], cost.loc[idx]
    R_M, c_M = R_M.loc[idx], c_M.loc[idx]

    cols = ret_m.columns
    out = {k: [] for k in ("ap_beta1", "ap_beta2", "ap_beta3", "ap_beta4", "ap_beta5")}
    dates = []

    W = AP_BETA_WINDOW_MONTHS
    for i in range(W - 1, len(idx)):
        sl = slice(i - W + 1, i + 1)
        Ri, Ci = ret_m.values[sl], cost.values[sl]
        Rm, Cm = R_M.values[sl], c_M.values[sl]

        denom = np.var(Rm - Cm, ddof=1)
        if not np.isfinite(denom) or denom <= 0:
            continue

        def _cov(Xn, y):
            """逐列 Cov(X[:,k], y)，按各列自己的非空行计算。"""
            M = ~np.isnan(Xn)
            n = M.sum(axis=0)
            X0 = np.where(M, Xn, 0.0)
            yb = (M * y[:, None]).sum(axis=0) / np.where(n > 0, n, np.nan)
            xb = X0.sum(axis=0) / np.where(n > 0, n, np.nan)
            cov = ((X0 - xb) * np.where(M, y[:, None] - yb, 0.0)).sum(axis=0)
            return np.where(n >= 24, cov / np.where(n > 1, n - 1, np.nan), np.nan)

        b1 = _cov(Ri, Rm) / denom
        b2 = _cov(Ci, Cm) / denom
        b3 = _cov(Ri, Cm) / denom
        b4 = _cov(Ci, Rm) / denom
        b5 = b2 - b3 - b4

        for k, v in zip(out, (b1, b2, b3, b4, b5)):
            out[k].append(pd.Series(v, index=cols))
        dates.append(idx[i])

    if not dates:
        return {}
    return {k: pd.DataFrame(v, index=pd.DatetimeIndex(dates)) for k, v in out.items()}


def _make_ap_beta_getter(key: str):
    """为 ap_beta1~5 各生成一个符合框架签名的因子函数。"""
    def _f(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        res = _ap_betas_raw(start=start, end=end)
        if not res or key not in res:
            return pd.DataFrame()
        return preprocess(res[key])
    _f.__name__ = f"_calc_{key}"
    _f.__doc__ = f"Acharya-Pedersen (2005) {key}，36 月滚动。见 _ap_betas_raw。"
    return _f


_calc_ap_beta1 = _make_ap_beta_getter("ap_beta1")
_calc_ap_beta2 = _make_ap_beta_getter("ap_beta2")
_calc_ap_beta3 = _make_ap_beta_getter("ap_beta3")
_calc_ap_beta4 = _make_ap_beta_getter("ap_beta4")
_calc_ap_beta5 = _make_ap_beta_getter("ap_beta5")


def _calc_capm_beta(start: Optional[str] = None,
                    end: Optional[str] = None) -> pd.DataFrame:
    """
    CAPM 市场 Beta（240 交易日滚动 OLS，月末取值）。

    r_i = α + β·r_m + ε

    用原始收益而非超额收益：rf 在截面上是常数，会被截距吸收，β 完全相同
    （原 factorcal.py FactorBeta 的注释也是这么说的）。
    窗口 CAPM_BETA_WINDOW(=240)，最少 MIN_ROLLING_VALID_DAYS 有效观测。

    闭式解 β = Cov(r_i, r_m) / Var(r_m)，与逐股 OLS 等价。

    需要数据：close_adj（日度）、ff3_daily（推市场收益）
    来源：[af] factorcal.py FactorBeta
    """
    close = load_data([_CLOSE], start=start, end=end).get(_CLOSE)
    if close is None:
        log.warning("[capm_beta] 缺少 close_adj 数据")
        return pd.DataFrame()

    mkt = _load_market_return_daily(start=start, end=end)
    if mkt is None:
        return pd.DataFrame()

    idx = close.index.intersection(mkt.index)
    ret, mkt = close.loc[idx].pct_change(), mkt.loc[idx]

    cols = ret.columns
    monthly_dates = pd.date_range(start=idx[0], end=idx[-1], freq="ME")
    rows, dates = [], []

    for month_end in monthly_dates:
        pos = idx <= month_end
        if pos.sum() < CAPM_BETA_WINDOW:
            continue
        sl = np.where(pos)[0][-CAPM_BETA_WINDOW:]

        Y = ret.values[sl]                       # T×N
        x = mkt.values[sl]                       # T
        good = ~np.isnan(x)
        Y, x = Y[good], x[good]

        M = ~np.isnan(Y)
        n = M.sum(axis=0)
        Y0 = np.where(M, Y, 0.0)
        # 每只股票用自己的非空行算均值，与逐股 dropna 后回归一致
        with np.errstate(invalid="ignore", divide="ignore"):
            xb = (M * x[:, None]).sum(axis=0) / np.where(n > 0, n, np.nan)
            yb = Y0.sum(axis=0) / np.where(n > 0, n, np.nan)
            xc = np.where(M, x[:, None] - xb, 0.0)
            cov = (xc * np.where(M, Y0 - yb, 0.0)).sum(axis=0)
            var = (xc ** 2).sum(axis=0)
            beta = np.where((n >= MIN_ROLLING_VALID_DAYS) & (var > 0),
                            cov / np.where(var > 0, var, np.nan), np.nan)

        rows.append(pd.Series(beta, index=cols))
        dates.append(month_end)

    if not dates:
        return pd.DataFrame()
    return preprocess(pd.DataFrame(rows, index=pd.DatetimeIndex(dates)))


def _rolling_ff3_ols(ret_excess: pd.DataFrame,
                     factors: pd.DataFrame,
                     min_window_days: int = 200,
                     min_stock_obs: int = 20) -> dict:
    """12 个月滚动 FF3 回归，**一次算完所有股票**（向量化）。

    对每个月末，取过去 12 个月的日度超额收益，逐股票拟合

        r_i^e = α + β_mkt·MKT + β_smb·SMB + β_hml·HML + ε

    一次返回 IVOL 与三个 Beta，供 `_calc_ivol` 和 `_calc_ff3_betas` 共用
    （原本两个函数各跑一遍同样的回归）。

    ── 为什么等价 ──────────────────────────────────────────────
    与逐股 `LinearRegression().fit()` 解的是同一组正规方程 (X'X)β = X'y，
    只是把「逐列求解」换成「批量求解」。缺失值按列各自处理：每只股票
    只用自己非空的那些交易日，与原实现的 `y.dropna()` 行为一致。
    用 `pinv`（SVD 最小二乘）而非 `solve`，与 sklearn 内部一致，
    对奇异/病态矩阵能优雅退化。

    ── 为什么要改 ──────────────────────────────────────────────
    原实现是 248 个月末 × ~3500 只股票的双重循环，两个因子合计约 174 万次
    sklearn 拟合，实测耗时约 30 分钟。此版约 10 秒。
    """
    idx  = ret_excess.index
    cols = ret_excess.columns
    n_stk = len(cols)

    monthly_dates = pd.date_range(start=idx[0], end=idx[-1], freq='ME')

    out = {k: [] for k in ('ivol', 'beta_mkt', 'beta_smb', 'beta_hml')}
    dates = []

    for month_end in monthly_dates:
        window_start = month_end - pd.DateOffset(months=11)
        wmask = (idx >= window_start) & (idx <= month_end)
        if wmask.sum() < min_window_days:
            continue

        F = factors.values[wmask]                    # T×3
        Y = ret_excess.values[wmask]                 # T×N

        # 因子本身缺失的交易日整行剔除（原实现中这类行会让 sklearn 报错并跳过该股）
        good = ~np.isnan(F).any(axis=1)
        F, Y = F[good], Y[good]
        T = len(F)
        if T < min_stock_obs:
            continue

        X  = np.column_stack([np.ones(T), F])        # T×4，第 0 列为截距
        M  = ~np.isnan(Y)                            # T×N 有效观测掩码
        Mf = M.astype(np.float64)
        Y0 = np.where(M, Y, 0.0)
        n_obs = M.sum(axis=0)                        # 每只股票的有效样本数

        # X'X 按各股票自己的缺失模式加权：X'diag(m_n)X
        # 用 16 次「矩阵×向量」而非三算子 einsum，内存占用更小
        XtX = np.empty((n_stk, 4, 4))
        for i in range(4):
            for j in range(i, 4):
                v = Mf.T @ (X[:, i] * X[:, j])
                XtX[:, i, j] = v
                XtX[:, j, i] = v
        Xty = (X.T @ Y0).T                           # N×4

        beta = np.full((n_stk, 4), np.nan)
        ok = n_obs >= min_stock_obs
        if ok.any():
            beta[ok] = np.einsum('nij,nj->ni',
                                 np.linalg.pinv(XtX[ok]), Xty[ok])

        resid = np.where(M, Y0 - X @ beta.T, 0.0)
        sse   = (resid ** 2).sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            ivol = np.sqrt(sse / (n_obs - 4)) * np.sqrt(252)
        ivol = np.where(ok, ivol, np.nan)

        s = lambda a: pd.Series(np.where(ok, a, np.nan), index=cols).dropna()
        out['ivol'].append(s(ivol))
        out['beta_mkt'].append(s(beta[:, 1]))
        out['beta_smb'].append(s(beta[:, 2]))
        out['beta_hml'].append(s(beta[:, 3]))
        dates.append(month_end)

    if not dates:
        return {}
    return {k: pd.DataFrame(v, index=dates) for k, v in out.items()}


@functools.lru_cache(maxsize=4)
def _ff3_rolling_result(start: Optional[str] = None,
                        end: Optional[str] = None) -> dict:
    """加载日度超额收益 + FF3，跑一次 `_rolling_ff3_ols`，结果按 (start,end) 缓存。

    ivol / ff3_betas / beta_smb / beta_hml 四个因子解的是同一组回归，
    共用此函数可避免在一次回测里重复跑四遍（每遍约 10 秒）。
    """
    from pathlib import Path

    close = load_data([_CLOSE], start=start, end=end).get(_CLOSE)
    if close is None:
        log.warning("[ff3_rolling] 缺少 close_adj 数据")
        return {}

    ff3_path = Path("/Users/louis/MyProjects/Database/data/factors/ff3_daily.csv")
    if not ff3_path.exists():
        log.warning(f"[ff3_rolling] FF3 文件不存在：{ff3_path}")
        return {}

    ff3 = pd.read_csv(ff3_path, index_col=0, parse_dates=True)[['MKT', 'SMB', 'HML', 'Rf']]

    ret_d = close.pct_change()
    idx = ret_d.index.intersection(ff3.index)
    ret_d, ff3 = ret_d.loc[idx], ff3.loc[idx]

    ret_excess = ret_d.sub(ff3['Rf'], axis=0)
    return _rolling_ff3_ols(ret_excess, ff3[['MKT', 'SMB', 'HML']].copy())


def _calc_ivol(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    特质波动率 IVOL（FF3回归残差年化标准差）。

    IVOL = √(SSE/(n-k)) × √252
    12月滚动窗口 FF3 回归，最少 MIN_ROLLING_VALID_DAYS 个有效交易日。

    回归模型：r_i^e = α + β_mkt×MKT + β_hml×HML + β_smb×SMB + ε
    其中 r_i^e = (r_i - rf) 为超额收益

    需要数据：close（日度）、FF3 日度因子（MKT/HML/SMB）、rf_daily
    来源：[af] factorcal.py FactorIVFF
    """
    from pathlib import Path

    # ── 加载日度收益率 ──────────────────────────────────────────
    data = load_data(["close_adj"], start=start, end=end)
    close = data.get("close_adj")
    if close is None:
        log.warning("[ivol] 缺少 close_adj 数据")
        return pd.DataFrame()

    ret_d = close.pct_change()

    # ── 加载 FF3 日度因子 ────────────────────────────────────────
    ff3_path = Path("/Users/louis/MyProjects/Database/data/factors/ff3_daily.csv")
    if not ff3_path.exists():
        log.warning(f"[ivol] FF3 文件不存在：{ff3_path}")
        return pd.DataFrame()

    ff3 = pd.read_csv(ff3_path, index_col=0, parse_dates=True)
    ff3_cols = ['MKT', 'SMB', 'HML', 'Rf']
    ff3 = ff3[ff3_cols]

    # ── 对齐日期索引 ────────────────────────────────────────────
    idx = ret_d.index.intersection(ff3.index)
    ret_d = ret_d.loc[idx]
    ff3 = ff3.loc[idx]

    # ── 计算超额收益 ────────────────────────────────────────────
    rf = ff3['Rf']
    ret_excess = ret_d.sub(rf, axis=0)   # 每列减去对应日期的 Rf

    # ── 准备因子矩阵（MKT, SMB, HML）────────────────────────────
    factors = ff3[['MKT', 'SMB', 'HML']].copy()

    # ── 月度滚动回归（12个月窗口，向量化，见 _rolling_ff3_ols）──────

    res = _rolling_ff3_ols(ret_excess, factors)
    if not res:
        return pd.DataFrame()

    return preprocess(res['ivol'])


def _calc_ff3_betas(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Fama-French 三因子 Beta（12月滚动 OLS）。

    回归模型：r_i^e = α + β_mkt×MKT + β_hml×HML + β_smb×SMB + ε
    其中 r_i^e = (r_i - rf) 为超额收益

    输出 DataFrame 包含三列：
      - beta_mkt: 市场 Beta（MKT 的回归系数）
      - beta_smb: 规模 Beta（SMB 的回归系数）
      - beta_hml: 价值 Beta（HML 的回归系数）

    注：返回值为合并 DataFrame，主因子为 beta_mkt（市场 Beta）

    需要数据：close（日度）、FF3 日度因子（MKT/HML/SMB）、rf_daily
    来源：[af] factorcal.py FactorBetaFF3
    """
    from pathlib import Path

    # ── 加载日度收益率 ──────────────────────────────────────────
    data = load_data(["close_adj"], start=start, end=end)
    close = data.get("close_adj")
    if close is None:
        log.warning("[ff3_betas] 缺少 close_adj 数据")
        return {}

    ret_d = close.pct_change()

    # ── 加载 FF3 日度因子 ────────────────────────────────────────
    ff3_path = Path("/Users/louis/MyProjects/Database/data/factors/ff3_daily.csv")
    if not ff3_path.exists():
        log.warning(f"[ff3_betas] FF3 文件不存在：{ff3_path}")
        return {}

    ff3 = pd.read_csv(ff3_path, index_col=0, parse_dates=True)
    ff3_cols = ['MKT', 'SMB', 'HML', 'Rf']
    ff3 = ff3[ff3_cols]

    # ── 对齐日期索引 ────────────────────────────────────────────
    idx = ret_d.index.intersection(ff3.index)
    ret_d = ret_d.loc[idx]
    ff3 = ff3.loc[idx]

    # ── 计算超额收益 ────────────────────────────────────────────
    rf = ff3['Rf']
    ret_excess = ret_d.sub(rf, axis=0)

    # ── 准备因子矩阵（MKT, SMB, HML）────────────────────────────
    factors = ff3[['MKT', 'SMB', 'HML']].copy()

    # ── 月度滚动回归（12个月窗口，向量化，见 _rolling_ff3_ols）──────

    res = _rolling_ff3_ols(ret_excess, factors)
    if not res:
        return pd.DataFrame()

    beta_mkt_df = res['beta_mkt']

    # 只返回 beta_mkt。SMB / HML 载荷改由下面两个独立因子提供
    # （原先塞进 result.attrs，而 attrs 在 to_csv/read_csv 缓存往返中会丢失，
    #   等于白算；2026-08-18 拆分）。
    return preprocess(beta_mkt_df)


def _calc_beta_smb(start: Optional[str] = None,
                   end: Optional[str] = None) -> pd.DataFrame:
    """
    Fama-French SMB 载荷（12 月滚动日度回归）。

    r_i^e = α + β_mkt·MKT + β_smb·SMB + β_hml·HML + ε 中的 β_smb。
    衡量个股对**规模因子**的暴露：高 → 表现更像小盘股。

    与直接的 size 因子不同——size 是市值水平（特征），β_smb 是对规模
    因子收益的协动（载荷）。GKX 两者都收，因为在 A 股未必同向。

    需要数据：close_adj（日度）、ff3_daily
    来源：[af] factorcal.py FactorBetaFF3
    """
    res = _ff3_rolling_result(start=start, end=end)
    if not res:
        return pd.DataFrame()
    return preprocess(res['beta_smb'])


def _calc_beta_hml(start: Optional[str] = None,
                   end: Optional[str] = None) -> pd.DataFrame:
    """
    Fama-French HML 载荷（12 月滚动日度回归）。

    衡量个股对**价值因子**的暴露：高 → 表现更像价值股。
    与 bm（账面市值比，特征）的区别同 beta_smb 与 size 的区别。

    需要数据：close_adj（日度）、ff3_daily
    来源：[af] factorcal.py FactorBetaFF3
    """
    res = _ff3_rolling_result(start=start, end=end)
    if not res:
        return pd.DataFrame()
    return preprocess(res['beta_hml'])


