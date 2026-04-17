# =============================================================================
# factors/fundamental.py
# 基本面因子库
#
# 包含所有财报/估值衍生的基本面信号：
#
#   pb               市净率（月末截面，行业+市值双重中性化）
#   pe_ttm           市盈率 TTM（月末截面，无中性化）
#   dividend_yield   股息率 TTM（月末截面，无中性化）
#   size             市值因子 log(流通市值)（月末截面，无中性化）
#   net_profit_yoy   净利润同比增速（季度，滞后3个季度，行业+市值双重中性化）
#
# 中性化说明（经用户确认）：
#   PB 和 NetProfit_YoY 做市值+行业双重中性化（延续 因子框架.py 做法）
#   PE、Dividend Yield、SIZE 暂不做中性化
#
# 数据来源标注：
#   [arch]   来自 _archive/raw_data/ 历史 CSV（2014-2020）
#   [db]     来自 Database/data/stock/A/（2021-至今）
#   [orig]   移植自原 因子框架.py
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional
import warnings

from data.loader import load_data, to_monthly
from data.universe import apply_universe, build_investable_mask
from factors.base import preprocess
from config.settings import (
    NP_YOY_LAG_QUARTERS,
    PB_NEUTRALIZE_SIZE,
    PB_NEUTRALIZE_INDUSTRY,
    MIN_ROLLING_VALID_DAYS,
    INDUSTRY_H5_PATH,
)


# -----------------------------------------------------------------------------
# 辅助：构建 log(流通市值) 用于中性化
# -----------------------------------------------------------------------------

def _get_log_mktcap(
    start: Optional[str],
    end: Optional[str],
    mask: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    计算月度 log(流通市值) = log(不复权收盘价 × 流通股本)。
    用于市值中性化。
    """
    data = load_data(["close", "float_shares"], start=start, end=end)
    close        = data.get("close")
    float_shares = data.get("float_shares")

    if close is None or float_shares is None:
        warnings.warn("[fundamental] 无法获取市值数据（close 或 float_shares 缺失）。")
        return None

    close_m        = to_monthly(apply_universe(close, mask),        method="last")
    float_shares_m = to_monthly(apply_universe(float_shares, mask), method="last")

    mktcap = (close_m * float_shares_m).replace(0, np.nan)
    return np.log(mktcap)


# -----------------------------------------------------------------------------
# 1. 市净率因子（PB）
#    来源：[arch] PB.csv / [db] pb.csv + [orig] 因子框架.py 中性化方法
# -----------------------------------------------------------------------------

def calc_pb(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市净率因子（PB）：月末截面值，越小代表越"价值"。
    经用户确认：做市值+行业双重中性化（延续 因子框架.py 做法）。

    数据：pb [arch: PB.csv] / [db: pb.csv]
    中性化：log(流通市值) + 行业哑变量（FactorLoading_Industry_arch.h5）
    """
    data = load_data(["pb", "close", "float_shares"], start=start, end=end)
    pb = data.get("pb")

    if pb is None:
        warnings.warn("[fundamental] calc_pb: 缺少 pb 数据")
        return pd.DataFrame()

    mask   = build_investable_mask(start=start, end=end, freq="D")
    pb_m   = to_monthly(apply_universe(pb, mask), method="last")

    # 确定中性化方式
    neutralize_mode = None
    log_mktcap = None

    if PB_NEUTRALIZE_SIZE and PB_NEUTRALIZE_INDUSTRY:
        neutralize_mode = "size+industry"
        log_mktcap = _get_log_mktcap(start, end, mask)
        if log_mktcap is None:
            neutralize_mode = "industry"  # 退化为仅行业中性化
    elif PB_NEUTRALIZE_SIZE:
        neutralize_mode = "size"
        log_mktcap = _get_log_mktcap(start, end, mask)
    elif PB_NEUTRALIZE_INDUSTRY:
        neutralize_mode = "industry"

    return preprocess(
        pb_m,
        neutralize=neutralize_mode,
        log_mktcap=log_mktcap,
        industry_h5_path=INDUSTRY_H5_PATH,
    )


# -----------------------------------------------------------------------------
# 2. 市盈率因子（PE TTM）
#    来源：[db] pe_ttm.csv
# -----------------------------------------------------------------------------

def calc_pe_ttm(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市盈率 TTM 因子：月末截面值，越小代表越"价值"。
    注：仅 Database（2021+）有此数据，存档期间不可用。
    暂不做中性化。

    数据：pe_ttm [db: pe_ttm.csv]
    """
    data = load_data(["pe_ttm"], start=start, end=end)
    pe = data.get("pe_ttm")

    if pe is None:
        warnings.warn("[fundamental] calc_pe_ttm: 缺少 pe_ttm 数据（仅 Database 2021+ 可用）")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    pe_m = to_monthly(apply_universe(pe, mask), method="last")

    # PE 可能存在负值（亏损公司），去极值时需特别注意
    # 此处将负 PE 置为 NaN（亏损股 PE 无意义作为估值因子）
    pe_m[pe_m <= 0] = np.nan

    return preprocess(pe_m)


# -----------------------------------------------------------------------------
# 3. 股息率因子（Dividend Yield TTM）
#    来源：[db] dividend_ratio.csv
# -----------------------------------------------------------------------------

def calc_dividend_yield(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    股息率 TTM 因子：月末截面值，越高代表股息越丰厚。
    注：仅 Database（2021+）有此数据。
    暂不做中性化。

    数据：dividend_ratio [db: dividend_ratio.csv]
    """
    data = load_data(["dividend_ratio"], start=start, end=end)
    div = data.get("dividend_ratio")

    if div is None:
        warnings.warn("[fundamental] calc_dividend_yield: 缺少 dividend_ratio 数据（仅 Database 2021+ 可用）")
        return pd.DataFrame()

    mask  = build_investable_mask(start=start, end=end, freq="D")
    div_m = to_monthly(apply_universe(div, mask), method="last")

    return preprocess(div_m)


# -----------------------------------------------------------------------------
# 4. 市值因子（SIZE）
#    来源：[af] factorcal.py FactorSIZE + [arch] close.csv & float_shares.csv
# -----------------------------------------------------------------------------

def calc_size(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市值因子：log(流通市值) = log(不复权收盘价 × 流通股本)，月末取值。
    市值越小，因子值越小（通常小盘股有超额收益）。
    暂不做中性化（市值因子本身是中性化的控制变量，不宜再自我中性化）。

    数据：close [arch]/[db]，float_shares [arch: 流通股本.csv]
    来源：[af] factorcal.py FactorSIZE
    """
    mask = build_investable_mask(start=start, end=end, freq="D")
    log_mktcap = _get_log_mktcap(start, end, mask)

    if log_mktcap is None:
        warnings.warn("[fundamental] calc_size: 无法计算 log(流通市值)")
        return pd.DataFrame()

    # SIZE 不做中性化，仅去极值和标准化
    return preprocess(log_mktcap, neutralize=None)


# -----------------------------------------------------------------------------
# 5. 净利润同比增速（NetProfit YoY Growth）
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
    - 去极值后做市值+行业双重中性化

    注：目前该数据仅存在于存档 CSV（归属母公司净利润.csv），
        Database 尚未覆盖净利润季度数据，后续补充后可扩展。

    数据：net_profit [arch: 归属母公司净利润.csv]（季度）
    年化频率调整：√3（季度数据，每年3个回测期）
    """
    data = load_data(["net_profit", "close", "float_shares"], start=start, end=end)
    net_profit = data.get("net_profit")

    if net_profit is None:
        warnings.warn("[fundamental] calc_net_profit_yoy: 缺少 net_profit 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")

    # 净利润为季度数据，对齐后计算同比增速
    # 存档 CSV 的净利润已是季度频率（index=季度末日期）
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

    # 市值+行业双重中性化（与 PB 一致）
    log_mktcap = _get_log_mktcap(start, end, mask)

    return preprocess(
        yoy,
        neutralize="size+industry",
        log_mktcap=log_mktcap,
        industry_h5_path=INDUSTRY_H5_PATH,
    )
