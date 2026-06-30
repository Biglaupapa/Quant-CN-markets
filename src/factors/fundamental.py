# =============================================================================
# factors/fundamental.py
# 基本面因子库
#
# 包含所有财报/估值衍生的基本面信号：
#
#   pb               市净率（月末截面，无中性化）
#   bm               账面市值比 log(1/PB)（月末截面，无中性化，正向因子）
#   pe_ttm           市盈率 TTM（静态，月末截面，无中性化）
#   pe1              动态市盈率（滚动12个月，月末截面，无中性化）
#   dividend_yield   股息率（近12个月，月末截面，无中性化）
#   size             市值因子 log(总市值)（月末截面，无中性化）
#                    → 数据源：market_value（Datayes，2004+）
#   size2            流通市值因子 log(流通市值)（月末截面，无中性化）
#                    → 数据源：neg_market_value（Datayes，2004+）
#   net_profit_yoy   净利润同比增速（季度，滞后3个季度，流通市值+行业双重中性化）
#
# 中性化说明：
#   net_profit_yoy 做流通市值+行业双重中性化
#   其余因子均不做中性化
#
# 数据来源标注：
#   [arch]   来自 _archive/raw_data/ 历史 CSV（2014-2020）
#   [db]     来自 Database/data/stock/A/（2004-至今）
#   [orig]   移植自原 因子框架.py
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional
import warnings

from src.data.loader import load_data, to_monthly
from src.data.universe import apply_universe, build_investable_mask
from src.factors.base import preprocess
from src.config.settings import (
    NP_YOY_LAG_QUARTERS,
    MIN_ROLLING_VALID_DAYS,
    INDUSTRY_H5_PATH,
)


# -----------------------------------------------------------------------------
# 辅助：构建 log(市值) 用于中性化与 Size 因子
# -----------------------------------------------------------------------------

def _get_log_mktcap(
    start: Optional[str],
    end: Optional[str],
    mask: pd.DataFrame,
    use_field: str = "neg_market_value",
) -> Optional[pd.DataFrame]:
    """
    计算月度 log(市值)，直接使用 Datayes 提供的市值字段。

    Args:
        use_field: "market_value"    （总市值，用于 Size 因子）
                   "neg_market_value"（流通市值，默认，用于 Size2 及市值中性化）

    数据覆盖：Datayes market_value / neg_market_value 均从 2004-01-02 起可用。
    """
    data = load_data([use_field], start=start, end=end)
    mv = data.get(use_field)

    if mv is None or mv.empty:
        warnings.warn(f"[fundamental] 无法获取市值数据：{use_field} 为空。")
        return None

    mv_m = to_monthly(apply_universe(mv, mask), method="last").replace(0, np.nan)
    if not mv_m.notna().any().any():
        warnings.warn(f"[fundamental] {use_field} 月度数据全为空。")
        return None

    return np.log(mv_m)


# -----------------------------------------------------------------------------
# 1. 市净率因子（PB）
#    来源：[arch] PB.csv / [db] pb.csv
# -----------------------------------------------------------------------------

def calc_pb(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市净率因子（PB）：月末截面值，越小代表越"价值"。
    无中性化。

    数据：pb [arch: PB.csv] / [db: pb.csv]
    """
    data = load_data(["pb"], start=start, end=end)
    pb = data.get("pb")

    if pb is None:
        warnings.warn("[fundamental] calc_pb: 缺少 pb 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    pb_m = to_monthly(apply_universe(pb, mask), method="last")

    return preprocess(pb_m)


# -----------------------------------------------------------------------------
# 2. 账面市值比因子（B/M）
#    来源：[arch] PB.csv / [db] pb.csv（取倒数后取对数）
# -----------------------------------------------------------------------------

def calc_bm(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    账面市值比因子（B/M = 1/PB）：log(1/PB) = -log(PB)。
    越大代表估值越低（价值股），正向因子（高 B/M → 高预期收益）。
    参考 Fama-French HML 构建逻辑，无中性化。

    数据：pb [arch: PB.csv] / [db: pb.csv]
    """
    data = load_data(["pb"], start=start, end=end)
    pb = data.get("pb")

    if pb is None:
        warnings.warn("[fundamental] calc_bm: 缺少 pb 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    pb_m = to_monthly(apply_universe(pb, mask), method="last")

    # PB <= 0 无意义，置 NaN
    pb_m[pb_m <= 0] = np.nan

    # B/M = log(1/PB) = -log(PB)
    bm_m = -np.log(pb_m)

    return preprocess(bm_m)


# -----------------------------------------------------------------------------
# 3. 市盈率因子（PE TTM）
#    来源：[db] pe_ttm.csv
# -----------------------------------------------------------------------------

def calc_pe_ttm(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市盈率 TTM 因子：月末截面值，越小代表越"价值"。
    负值（亏损公司）置为 NaN。无中性化。

    数据：pe_ttm [db: pe_ttm.csv]（Datayes，2004+）
    """
    data = load_data(["pe_ttm"], start=start, end=end)
    pe = data.get("pe_ttm")

    if pe is None:
        warnings.warn("[fundamental] calc_pe_ttm: 缺少 pe_ttm 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    pe_m = to_monthly(apply_universe(pe, mask), method="last")
    pe_m[pe_m <= 0] = np.nan

    return preprocess(pe_m)


# -----------------------------------------------------------------------------
# 4. 动态市盈率因子（PE1）
#    来源：[db] pe1.csv（Datayes getMktEqud.PE1）
# -----------------------------------------------------------------------------

def calc_pe1(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    动态市盈率因子（PE1）：月末截面值，越小代表估值越低。

    PE1 与 pe_ttm 的区别：
      pe_ttm = 总市值 / 最近一年报告期净利润（静态，基于已披露年报）
      pe1    = 总市值 / 滚动12个月盈利（动态，含最新季报，更及时）

    负值（亏损公司）置为 NaN。无中性化。

    数据：pe1 [db: pe1.csv]（Datayes，2004+）
    """
    data = load_data(["pe1"], start=start, end=end)
    pe1 = data.get("pe1")

    if pe1 is None:
        warnings.warn("[fundamental] calc_pe1: 缺少 pe1 数据")
        return pd.DataFrame()

    mask  = build_investable_mask(start=start, end=end, freq="D")
    pe1_m = to_monthly(apply_universe(pe1, mask), method="last")
    pe1_m[pe1_m <= 0] = np.nan

    return preprocess(pe1_m)


# -----------------------------------------------------------------------------
# 5. 股息率因子（Dividend Yield）
#    来源：[db] dividend_ratio.csv
# -----------------------------------------------------------------------------

def calc_dividend_yield(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    股息率 TTM 因子：月末截面值，越高代表股息越丰厚。
    无中性化。

    数据：dividend_ratio [db: dividend_ratio.csv]（Datayes，2004+）
    """
    data = load_data(["dividend_ratio"], start=start, end=end)
    div = data.get("dividend_ratio")

    if div is None:
        warnings.warn("[fundamental] calc_dividend_yield: 缺少 dividend_ratio 数据")
        return pd.DataFrame()

    mask  = build_investable_mask(start=start, end=end, freq="D")
    div_m = to_monthly(apply_universe(div, mask), method="last")

    return preprocess(div_m)


# -----------------------------------------------------------------------------
# 6. 市值因子（SIZE）— 总市值
#    数据：market_value（Datayes getMktDivYield，2004+）
# -----------------------------------------------------------------------------

def calc_size(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Size 因子：log(总市值)，月末取值。
    不做中性化（市值因子本身是中性化的控制变量，不宜自我中性化）。

    数据：market_value（Datayes，2004+）
    """
    mask = build_investable_mask(start=start, end=end, freq="D")
    log_mktcap = _get_log_mktcap(start, end, mask, use_field="market_value")

    if log_mktcap is None:
        warnings.warn("[fundamental] calc_size: 无法计算 log(总市值)")
        return pd.DataFrame()

    return preprocess(log_mktcap, neutralize=None)


# -----------------------------------------------------------------------------
# 7. 流通市值因子（SIZE2）— 流通市值
#    数据：neg_market_value（Datayes getMktEqud，2004+）
# -----------------------------------------------------------------------------

def calc_size2(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Size2 因子：log(流通市值)，月末取值。
    与 Size（总市值）的差异体现在限售股比例较高的个股上。
    不做中性化。

    数据：neg_market_value（Datayes，2004+）
    """
    mask = build_investable_mask(start=start, end=end, freq="D")
    log_mktcap = _get_log_mktcap(start, end, mask, use_field="neg_market_value")

    if log_mktcap is None:
        warnings.warn("[fundamental] calc_size2: 无法计算 log(流通市值)")
        return pd.DataFrame()

    return preprocess(log_mktcap, neutralize=None)


# -----------------------------------------------------------------------------
# 8. 净利润同比增速（NetProfit YoY Growth）
#    来源：[arch] 归属母公司净利润.csv + [orig] 因子框架.py
# -----------------------------------------------------------------------------

def calc_net_profit_yoy(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    净利润同比增速：当季净利润 / 上年同季净利润 - 1。

    处理细节（延续 因子框架.py 做法）：
    - 数据为季度频率，滞后 NP_YOY_LAG_QUARTERS 个季度（财报披露延迟，默认3季度）
    - 即：在 t 月使用 t-3 季度的财务数据，确保无前瞻偏差
    - 去极值后做流通市值+行业双重中性化

    注：目前该数据仅存在于存档 CSV（归属母公司净利润.csv），
        Database 尚未覆盖净利润季度数据，后续补充后可扩展。

    数据：net_profit [arch: 归属母公司净利润.csv]（季度）
    """
    data = load_data(["net_profit"], start=start, end=end)
    net_profit = data.get("net_profit")

    if net_profit is None:
        warnings.warn("[fundamental] calc_net_profit_yoy: 缺少 net_profit 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")

    # 净利润为季度数据，对齐后计算同比增速
    np_monthly = net_profit  # 保持原始季度频率

    # 应用股票池掩码（月末掩码对齐）
    mask_m = mask.resample("ME").last().fillna(False)

    # 同比增速：当期 / 上年同期 - 1（即 shift(4) 对季度数据）
    yoy = np_monthly / np_monthly.shift(4) - 1

    # 滞后 NP_YOY_LAG_QUARTERS 个季度（财报披露延迟）
    yoy = yoy.shift(NP_YOY_LAG_QUARTERS)

    # 去除极端值（同比增速可能因基期接近零而出现极大值）
    yoy = yoy.replace([np.inf, -np.inf], np.nan)

    # 月度对齐（季度末填充到月度）
    yoy = yoy.resample("ME").last().ffill(limit=2)

    # 应用股票池掩码
    common_idx  = yoy.index.intersection(mask_m.index)
    common_cols = yoy.columns.intersection(mask_m.columns)
    yoy = yoy.reindex(index=common_idx, columns=common_cols)
    mask_aligned = mask_m.reindex(index=common_idx, columns=common_cols)
    yoy[~mask_aligned] = np.nan

    # 流通市值+行业双重中性化
    log_mktcap = _get_log_mktcap(start, end, mask, use_field="neg_market_value")

    return preprocess(
        yoy,
        neutralize="size+industry",
        log_mktcap=log_mktcap,
        industry_h5_path=INDUSTRY_H5_PATH,
    )
