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

from src.data.loader import load_data, load_data_hk, to_monthly
from src.config.settings import IPO_FILTER_DAYS, PENNY_STOCK_PRICE_MIN


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
        # 兼容两种格式：
        #   存档：字符串 "1" / "1.0" = ST
        #   Database：数值 1.0 = ST
        st_numeric = pd.to_numeric(st.stack(), errors="coerce").unstack()
        mask &= (st_numeric != 1)

    # --- 规则 2：剔除停牌股 ---
    if trade_status is not None:
        ts = trade_status.reindex(columns=all_cols)
        # 兼容两种格式（数据中可能混合存在，逐格判断）：
        #   存档：字符串 "交易" = 正常交易
        #   Database：数值 1.0 = 正常交易，0.0 = 停牌
        # 注意：不能用 if/else 分支，因为 2004-2021 间
        #   2004-2013 / 2021+ 为数值，2014-2021 为字符串（存档优先），
        #   必须逐格兼容，否则字符串行全部变 NaN 被误判为停牌。
        ts_numeric = pd.to_numeric(ts.stack(), errors="coerce").unstack()
        tradeable  = (ts_numeric == 1) | (ts == "交易")
        mask &= tradeable

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
# 港股专用：生成可投资掩码（仅依赖 close_adj）
# -----------------------------------------------------------------------------

def build_investable_mask_hk(
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "D",
) -> pd.DataFrame:
    """
    港股股票池可投资掩码（日度或月度）。

    由于港股没有 A 股的 ST/status/listed_days 等专用状态字段，
    仅依赖 close_adj 推断以下三项：

    规则 1：交易日过滤
        close_adj 有非 NaN 值 → 正常交易日（等价于 A 股 status==1）

    规则 2：仙股过滤（等价于 A 股 ST 剔除）
        close_adj >= PENNY_STOCK_PRICE_MIN（默认 HKD 1.0）
        港交所对长期低于 0.5 港元的股票有退市机制，HKD 1.0 是保守阈值。

    规则 3：次新股过滤（等价于 A 股 listed_days >= 60）
        累计有效交易日数 >= IPO_FILTER_DAYS（默认 60 日）
        使用 close_adj.notna().cumsum()，从有记录之日起累计。

    Parameters
    ----------
    start, end : str, optional
        时间范围，格式 "YYYY-MM-DD"
    freq : str
        "D"=日度掩码，"M"=月度掩码（取月末值）

    Returns
    -------
    pd.DataFrame
        布尔 DataFrame，index=DatetimeIndex，columns=港股代码
        True=该股票在该日期可投资，False=需剔除
    """
    # 读取后复权收盘价（港股唯一状态代理变量）
    # ── 关键：cumsum 必须从数据最早日期（2004-01-02）开始计算，
    #    不能传入 start，否则 2007 年回测起点附近的新上市股票会
    #    因累计天数不足 60 而被错误剔除。
    #    计算完 cumsum 后，再把 mask 裁剪回 start/end 范围。
    data = load_data_hk(["close_adj"], end=end)   # 不传 start，加载全量历史
    close_adj = data.get("close_adj")

    if close_adj is None or close_adj.empty:
        raise ValueError("[universe_hk] 无法加载港股 close_adj，无法构建股票池掩码")

    # 规则 1：正常交易日（有数据）
    trading = close_adj.notna()

    # 规则 2：非仙股（股价 >= HKD 1.0）
    not_penny = close_adj >= PENNY_STOCK_PRICE_MIN

    # 规则 3：非次新股（累计有效交易日 >= IPO_FILTER_DAYS）
    # cumsum 从 2004-01-02 开始全量计算，保证 2007 年起点时各股已有足够历史
    cum_trading_days = trading.cumsum()
    not_new_listing  = cum_trading_days >= IPO_FILTER_DAYS

    mask = trading & not_penny & not_new_listing

    # 裁剪至请求的时间范围（cumsum 已在全量历史上计算，裁剪不影响结果）
    if start:
        mask = mask.loc[start:]
    if end:
        mask = mask.loc[:end]

    if freq == "M":
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
