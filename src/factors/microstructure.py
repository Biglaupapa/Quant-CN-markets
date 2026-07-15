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
# ── 待激活因子（需额外数据，暂注释掉调用）────────────────────────────────
#   ps_gamma         Pastor-Stambaugh Gamma（需市场收益率日序列）
#   ps_liq_beta      PS 流动性 Beta（需先算 Gamma，36月滚动）
#   ap_betas         Acharya-Pedersen β1-β5（需市值、成交额）
#   capm_beta        CAPM 市场 Beta（240日滚动，需市场收益率）
#   ivol             特质波动率（FF3残差年化标准差，需FF3因子+无风险利率）
#   ff3_betas        Fama-French 三因子 Beta（需FF3因子+无风险利率）
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
import warnings
from sklearn.linear_model import LinearRegression

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

    for period_end, group in ret.resample("ME"):
        if len(group) < MIN_ROLLING_VALID_DAYS:
            frames.append(pd.Series(np.nan, index=group.columns))
            dates.append(period_end)
            continue

        # 协方差：r_t 与 r_{t-1}
        cov_row = {}
        for col in group.columns:
            s = group[col].dropna()
            if len(s) < MIN_ROLLING_VALID_DAYS:
                cov_row[col] = np.nan
                continue
            cov = np.cov(s.values[1:], s.values[:-1])[0, 1]
            # Cov < 0：bid-ask bounce 可检测，Roll spread 有意义，赋正值
            # Cov ≥ 0：Roll 公式产生虚数，无法估计 spread，置 NaN（不是 0）
            # 理由：Fong(2017) 和 Goyenko(2009) 均指出，正样本自相关
            #   对应真实 serial correlation 接近零的高流动性股票，
            #   Roll 模型对这类股票没有判别力。
            #   文献习惯赋 0（用于流动性水平比较），但在截面因子排序中，
            #   大量 0 值聚集会导致 pd.qcut 分组崩溃（57% 数据同值）。
            #   改为 NaN：明确区分"无法估计"与"流动性充分"，
            #   在 Cov<0 的子集（约 43% 股票-月份）内部排序，经济含义清晰。
            cov_row[col] = 2 * np.sqrt(-cov) if cov < 0 else np.nan

        frames.append(pd.Series(cov_row))
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
# ── 待激活因子（需额外数据，暂不调用）─────────────────────────────────────
# 以下函数已完整实现（移植自 AF-pricing/Coding/factorcal.py），
# 但需要 Database 中补充以下数据后才能激活：
#   - 市场日度收益率序列（marketrtn_daily）
#   - FF3 日度因子（RiskPremium, HML, SMB）
#   - 无风险利率日度序列（rf_daily）
# 激活方式：在 main() 中将对应 FACTOR_FLAGS 中的布林值改为 True。
# =============================================================================

def _calc_ps_gamma(data: dict) -> pd.DataFrame:
    """
    [待激活] Pastor-Stambaugh (2003) Gamma 流动性指标。

    月度截面回归：r^e_{i,d+1} = θ + φ·r_{i,d} + γ·sign(r^e_{i,d})·vol_{i,d} + ε
    其中 r^e = r_i - r_market（超额收益），vol = 成交额（百万元）。
    γ 为流动性系数，值越负表示流动性越差。

    需要数据：close（日度）、amt（千元）、marketrtn_daily（日度市场收益率）
    来源：[af] factorcal.py FactorGamma
    """
    # TODO: 激活条件 - Database 中补充 marketrtn_daily.csv
    raise NotImplementedError(
        "[microstructure] PS Gamma 尚未激活。"
        "请在 Database 中补充 marketrtn_daily 数据后在 main() 中启用。"
    )


def _calc_ps_liq_beta(gamma: pd.DataFrame, mktcap: pd.DataFrame) -> pd.DataFrame:
    """
    [待激活] Pastor-Stambaugh (2003) 流动性 Beta（36月滚动）。

    Step1: 聚合流动性水平 γ_t（市值加权）
    Step2: 流动性变化 Δγ_t
    Step3: AR 回归创新项 Lm（流动性冲击）
    Step4: 36月滚动回归 r_i ~ r_m + Lm，取 Lm 系数为 PSL Beta

    需要数据：PS Gamma 序列、市值序列、市场收益率
    来源：[af] factorcal.py FactorLiquidityBeta
    """
    raise NotImplementedError(
        "[microstructure] PS Liquidity Beta 尚未激活。"
        "需先激活 PS Gamma，并在 Database 中补充市场收益率数据。"
    )


def _calc_ap_betas(data: dict) -> Dict[str, pd.DataFrame]:
    """
    [待激活] Acharya-Pedersen (2005) 五个流动性 Beta（β1-β5，36月滚动）。

    β1: Cov(R_i, R_M) / Var(R_M - C_M)        市场收益 Beta
    β2: Cov(C_i, C_M) / Var(R_M - C_M)        流动性-流动性 Beta
    β3: Cov(R_i, C_M) / Var(R_M - C_M)        收益-流动性 Beta
    β4: Cov(C_i, R_M) / Var(R_M - C_M)        流动性-收益 Beta
    β5: β2 - β3 - β4                           净流动性 Beta

    需要数据：close（日度）、amt（千元）、marketvalue（日度市值）
    来源：[af] factorcal.py FactorLiquidityBetaAP
    参数：c1=0.25, c2=0.30, c3=60（illiquidity cost 上限）
    """
    raise NotImplementedError(
        "[microstructure] AP Betas 尚未激活。"
        "请在 Database 中补充 marketvalue（流通市值日度序列）后启用。"
    )


def _calc_capm_beta(data: dict) -> pd.DataFrame:
    """
    [待激活] CAPM 市场 Beta（240日滚动 OLS，月末取值）。

    r_i = α + β × r_m + ε，β 为市场 Beta，最少需 200 个有效观测。

    需要数据：close（日度）、marketrtn_daily（日度市场收益率）
    来源：[af] factorcal.py FactorBeta
    """
    raise NotImplementedError(
        "[microstructure] CAPM Beta 尚未激活。"
        "请在 Database 中补充 marketrtn_daily 数据后启用。"
    )


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
    data = load_data(start=start, end=end)
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

    # ── 月度滚动回归（12个月窗口） ──────────────────────────────
    monthly_dates = pd.date_range(
        start=ret_excess.index[0],
        end=ret_excess.index[-1],
        freq='ME'
    )

    ivol_series = []
    ivol_dates = []

    for month_end in monthly_dates:
        # 12月滚动窗口：[month_end - 11 months, month_end]
        window_start = month_end - pd.DateOffset(months=11)
        window = ret_excess.index[(ret_excess.index >= window_start) & (ret_excess.index <= month_end)]

        if len(window) < 200:  # 最少200个交易日
            continue

        ret_window = ret_excess.loc[window]
        factors_window = factors.loc[window]

        # 对每个股票做 FF3 回归
        ivol_month = {}
        for stock in ret_window.columns:
            y = ret_window[stock].dropna()
            X = factors_window.loc[y.index].copy()

            # 添加常数项
            X['const'] = 1.0
            X = X[['const', 'MKT', 'SMB', 'HML']]

            if len(y) < 20:  # 该窗口内该股票数据不足
                continue

            try:
                # OLS 回归
                model = LinearRegression()
                model.fit(X.iloc[:, 1:], y)  # 不用 const，sklearn 自动添加

                residuals = y - (model.intercept_ + model.predict(X.iloc[:, 1:]))

                # IVOL = √(SSE/(n-k)) × √252，k=4（alpha + 3 betas）
                n = len(residuals)
                sse = (residuals ** 2).sum()
                ivol = np.sqrt(sse / (n - 4)) * np.sqrt(252)

                ivol_month[stock] = ivol
            except Exception as e:
                continue

        if ivol_month:
            ivol_series.append(pd.Series(ivol_month))
            ivol_dates.append(month_end)

    if not ivol_series:
        return pd.DataFrame()

    ivol_df = pd.DataFrame(ivol_series, index=ivol_dates)
    return preprocess(ivol_df)


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
    data = load_data(start=start, end=end)
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

    # ── 月度滚动回归（12个月窗口） ──────────────────────────────
    monthly_dates = pd.date_range(
        start=ret_excess.index[0],
        end=ret_excess.index[-1],
        freq='ME'
    )

    beta_mkt_series = []
    beta_smb_series = []
    beta_hml_series = []
    beta_dates = []

    for month_end in monthly_dates:
        # 12月滚动窗口
        window_start = month_end - pd.DateOffset(months=11)
        window = ret_excess.index[(ret_excess.index >= window_start) & (ret_excess.index <= month_end)]

        if len(window) < 200:
            continue

        ret_window = ret_excess.loc[window]
        factors_window = factors.loc[window]

        # 对每个股票做 FF3 回归
        betas_month_mkt = {}
        betas_month_smb = {}
        betas_month_hml = {}

        for stock in ret_window.columns:
            y = ret_window[stock].dropna()
            X = factors_window.loc[y.index].copy()

            if len(y) < 20:
                continue

            try:
                # OLS 回归
                model = LinearRegression()
                model.fit(X, y)

                # 提取三个因子的 Beta（回归系数）
                # model.coef_ 顺序为 ['MKT', 'SMB', 'HML']
                betas_month_mkt[stock] = model.coef_[0]
                betas_month_smb[stock] = model.coef_[1]
                betas_month_hml[stock] = model.coef_[2]
            except Exception as e:
                continue

        if betas_month_mkt:
            beta_mkt_series.append(pd.Series(betas_month_mkt))
            beta_smb_series.append(pd.Series(betas_month_smb))
            beta_hml_series.append(pd.Series(betas_month_hml))
            beta_dates.append(month_end)

    if not beta_mkt_series:
        return pd.DataFrame()

    # ── 转换为 DataFrame 并预处理 ──────────────────────────────
    beta_mkt_df = pd.DataFrame(beta_mkt_series, index=beta_dates)
    beta_smb_df = pd.DataFrame(beta_smb_series, index=beta_dates)
    beta_hml_df = pd.DataFrame(beta_hml_series, index=beta_dates)

    # 合并三个 beta 为单个 DataFrame（作为多列因子）
    # 主列为 beta_mkt（市场 Beta），其他两列为辅助
    combined = pd.concat([
        beta_mkt_df.rename(columns=lambda x: f'beta_mkt_{x}'),
        beta_smb_df.rename(columns=lambda x: f'beta_smb_{x}'),
        beta_hml_df.rename(columns=lambda x: f'beta_hml_{x}'),
    ], axis=1)

    # 取第一个 beta（MKT）作为主因子返回，用于标准回测流程
    # 其他 beta 保留在 DataFrame 的其他列中供选择性使用
    result = preprocess(beta_mkt_df)
    # 附加其他列供后续参考
    result.attrs['beta_smb'] = beta_smb_df
    result.attrs['beta_hml'] = beta_hml_df

    return result


