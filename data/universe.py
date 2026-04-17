# =============================================================================
# data/universe.py
# 股票池定义与数据清洗（合并）
#
# 职责：
#   1. 加载过滤所需的状态数据（ST、停牌、上市日数）
#   2. 生成 investable_mask：布尔 DataFrame，True=可投资，False=剔除
#   3. 将 mask 应用到任意因子/价格 DataFrame（将不可投资的值置 NaN）
#
# 过滤规则（全部经用户确认）：
#   - 剔除 ST / *ST 股票
#   - 剔除停牌股（交易状态 != "交易"）
#   - 剔除次新股（上市不足 IPO_FILTER_DAYS 个交易日）
#
# 注意：年度交易日阈值暂不启用；
#       滚动窗口内最少 MIN_ROLLING_VALID_DAYS 个有效日由各因子内部控制。
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional

from data.loader import load_data, to_monthly
from config.settings import IPO_FILTER_DAYS


# -----------------------------------------------------------------------------
# 核心：生成可投资掩码
# -----------------------------------------------------------------------------

def build_investable_mask(
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "D",
) -> pd.DataFrame:
    """
    生成股票池可投资掩码（日度或月度）。

    Parameters
    ----------
    start, end : str, optional
        时间范围，格式 "YYYY-MM-DD"
    freq : str
        "D"=日度掩码，"M"=月度掩码（取月末值）

    Returns
    -------
    pd.DataFrame
        布尔 DataFrame，index=DatetimeIndex，columns=股票代码
        True=该股票在该日期可投资，False=需剔除
    """
    data = load_data(
        fields=["is_st", "trade_status", "listing_days"],
        start=start,
        end=end,
    )

    is_st        = data.get("is_st")
    trade_status = data.get("trade_status")
    listing_days = data.get("listing_days")

    # 对齐股票代码（columns 取交集，避免因来源不同导致维度不一致）
    all_cols = None
    for df in [is_st, trade_status, listing_days]:
        if df is not None:
            all_cols = df.columns if all_cols is None else all_cols.intersection(df.columns)

    # 初始化：默认全部可投资
    reference = next(df for df in [is_st, trade_status, listing_days] if df is not None)
    mask = pd.DataFrame(True, index=reference.index, columns=all_cols)

    # --- 规则 1：剔除 ST 股 ---
    if is_st is not None:
        st = is_st.reindex(columns=all_cols)
        # 值为 1 表示 ST；部分数据源用字符串"1"，做兼容处理
        mask &= (st.astype(str).replace({"1.0": "1", "nan": "0"}) != "1")

    # --- 规则 2：剔除停牌股 ---
    if trade_status is not None:
        ts = trade_status.reindex(columns=all_cols)
        mask &= (ts == "交易")

    # --- 规则 3：剔除次新股 ---
    if listing_days is not None:
        ld = listing_days.reindex(columns=all_cols)
        # listing_days 可能为字符串，需转 float
        ld_numeric = pd.to_numeric(ld.stack(), errors="coerce").unstack()
        mask &= (ld_numeric >= IPO_FILTER_DAYS)

    # --- 降频至月度 ---
    if freq == "M":
        # 月末取值：如果当月末该股票不可投资，则整月剔除
        mask = mask.resample("ME").last().fillna(False)

    return mask


# -----------------------------------------------------------------------------
# 应用掩码：将不可投资位置置 NaN
# -----------------------------------------------------------------------------

def apply_universe(
    factor: pd.DataFrame,
    mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    将可投资掩码应用到任意因子/价格 DataFrame。

    Parameters
    ----------
    factor : pd.DataFrame
        待过滤的数据，index=DatetimeIndex，columns=股票代码
    mask : pd.DataFrame
        由 build_investable_mask() 生成，与 factor 的频率一致

    Returns
    -------
    pd.DataFrame
        同 factor 形状，不可投资位置替换为 NaN
    """
    # 对齐时间和股票两个维度
    common_idx  = factor.index.intersection(mask.index)
    common_cols = factor.columns.intersection(mask.columns)

    result = factor.reindex(index=common_idx, columns=common_cols).copy()
    m      = mask.reindex(index=common_idx, columns=common_cols)

    result[~m] = np.nan
    return result


# -----------------------------------------------------------------------------
# 便捷函数：一步完成加载 + 过滤
# -----------------------------------------------------------------------------

def get_clean_field(
    field: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "D",
) -> pd.DataFrame:
    """
    加载字段并直接应用股票池过滤。

    Parameters
    ----------
    field : str
        字段名，参见 loader.load_field()
    start, end : str, optional
        时间范围
    freq : str
        "D"=日度，"M"=月度

    Returns
    -------
    pd.DataFrame
        过滤后的干净数据，不可投资位置为 NaN
    """
    from data.loader import load_field, to_monthly

    raw  = load_field(field, start=start, end=end)
    mask = build_investable_mask(start=start, end=end, freq="D")

    if freq == "M":
        raw  = to_monthly(raw)
        mask = mask.resample("ME").last().fillna(False)

    return apply_universe(raw, mask)
