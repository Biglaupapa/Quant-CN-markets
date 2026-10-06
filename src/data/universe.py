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
from src.config.settings import (IPO_FILTER_DAYS, PENNY_STOCK_PRICE_MIN, FORMATION_CONFIG,
                                 STOCKS_LIST_PATH, DELIST_PERIOD_PATH)


# -----------------------------------------------------------------------------
# 组建日（t 月末）股票池 —— 2026-10-01 起与文献对齐
# -----------------------------------------------------------------------------
#
# 与 build_investable_mask 的分工：
#   build_investable_mask(freq="D")  逐日清洗掩码，只用于**因子计算输入**（不变）
#   build_formation_mask()           t 月末组建股票池，决定**谁能入组**（engine.calc_monthly_returns 用）
#
# 规则、依据与实测见 docs/【方法】回测口径与文献对齐.md；参数见 settings.FORMATION_CONFIG。

def _month_end_rows(df: pd.DataFrame) -> pd.DataFrame:
    """取每月**最后一个交易日**那一行（不是「当月最后非空值」），索引换成日历月末。"""
    idx = df.index
    last = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
    out = df.loc[last.values]
    out.index = pd.DatetimeIndex(last.values) + pd.offsets.MonthEnd(0)
    return out


def formation_market_cap(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """t 月末 A 股市值（财汇 `a_market_value` = TOTMKTCAP，含限售股），取每月最后一个交易日那一行。

    全框架「市值」的唯一口径：规则 g（剔最小 30%）、组合市值加权（#40）共用本函数；
    LSY 复现 / CH 因子亦为 A 股市值（见方法文档 §十二）。
    """
    d = load_data(["a_market_value"], start=start, end=end)
    cap = pd.to_numeric(d["a_market_value"].stack(), errors="coerce").unstack()
    return _month_end_rows(cap)


def _in_sample(codes: pd.Index, sample: str) -> pd.Series:
    """a：样本范围。lsy = 60/00/30；lsy_star = + 688/689；all = 全部（含北交所）。"""
    is_bj = codes.str.endswith(".BJ")
    is_star = codes.str[:3].isin(["688", "689"])
    if sample == "all":
        keep = np.ones(len(codes), dtype=bool)
    elif sample == "lsy_star":
        keep = ~is_bj
    elif sample == "lsy":
        keep = ~is_bj & ~is_star & codes.str[:2].isin(["60", "00", "30"])
    else:
        raise ValueError(f"[universe] 未知 sample={sample!r}（lsy / lsy_star / all）")
    return pd.Series(keep, index=codes)


def build_formation_mask(
    start: Optional[str] = None,
    end: Optional[str] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    t 月末组建股票池（布尔，index = 日历月末，columns = 股票代码）。只用 t 月及以前信息。

    config 覆盖 settings.FORMATION_CONFIG 的任意键；值为 None / False 即关闭该规则。
    额外键 `min_listed_days`（默认不启用）：旧规则「listed_days ≥ N 交易日」，仅供新旧对比。
    """
    cfg = {**FORMATION_CONFIG, **(config or {})}
    # 往前多取 13 个月，供「过去 12 个月成交天数」使用
    load_start = (pd.Timestamp(start) - pd.DateOffset(months=13)).strftime("%Y-%m-%d") if start else None
    fields = (["trade_status", "is_st"]
              + (["listing_days"] if cfg.get("min_listed_days") else []))
    d = load_data(fields, start=load_start, end=end)
    num = lambda x: pd.to_numeric(x.stack(), errors="coerce").unstack()
    status = num(d["trade_status"])
    cols = status.columns
    traded = (status == 1)

    st_me = _month_end_rows(status)
    mask = pd.DataFrame(True, index=st_me.index, columns=cols)

    # b：t 月最后交易日有成交
    if cfg.get("trading_on_formation"):
        mask &= (st_me == 1)
    # c：交易记录天数（当月 / 过去 12 个月）
    days_m = traded.resample("ME").sum().reindex(index=mask.index, columns=cols)
    if cfg.get("min_trade_days_month"):
        mask &= days_m >= cfg["min_trade_days_month"]
    if cfg.get("min_trade_days_12m"):
        days_12 = days_m.rolling(12, min_periods=1).sum()
        mask &= days_12 >= cfg["min_trade_days_12m"]
    # d：上市满 N 个月（日历）
    if cfg.get("min_listed_months"):
        sl = pd.read_csv(STOCKS_LIST_PATH, dtype=str).set_index("code")
        ld = pd.to_datetime(sl["list_date"].reindex(cols), errors="coerce")
        ok_from = ld + pd.DateOffset(months=cfg["min_listed_months"])
        mask &= pd.DataFrame(mask.index.values[:, None] >= ok_from.values[None, :],
                             index=mask.index, columns=cols)
    # 旧规则（仅对比用）：listed_days ≥ N
    if cfg.get("min_listed_days"):
        ld_me = _month_end_rows(num(d["listing_days"])).reindex(index=mask.index, columns=cols)
        mask &= ld_me >= cfg["min_listed_days"]
    # e：t 月最后交易日 ST / *ST
    if cfg.get("exclude_st"):
        mask &= ~(_month_end_rows(num(d["is_st"])).reindex(index=mask.index, columns=cols) == 1)
    # f：待退市（退市整理期）
    if cfg.get("exclude_delist_period"):
        dp = pd.read_csv(DELIST_PERIOD_PATH, dtype=str)
        dp["me"] = pd.to_datetime(dp["month_end"]) + pd.offsets.MonthEnd(0)
        hit = pd.crosstab(dp["me"], dp["code"]).astype(bool).reindex(index=mask.index, columns=cols,
                                                                     fill_value=False)
        mask &= ~hit
    # a：样本范围
    if cfg.get("sample"):
        mask &= _in_sample(cols, cfg["sample"]).values[None, :]
    # g：A 股市值（财汇 TOTMKTCAP，含限售股）最小 X% 剔除（排序范围：所选样本内当月有市值的全部股票）
    if cfg.get("exclude_bottom_size"):
        cap = formation_market_cap(load_start, end).reindex(index=mask.index, columns=cols)   # 与组合市值加权同口径
        if cfg.get("sample"):
            keep = _in_sample(cols, cfg["sample"]).values
            cap = cap.loc[:, keep]                       # 排序范围限定在样本内
        small = (cap.rank(axis=1, pct=True) <= cfg["bottom_size_pct"]).reindex(columns=cols, fill_value=False)
        mask &= ~small

    mask = mask.fillna(False).astype(bool)
    if start:
        mask = mask.loc[pd.Timestamp(start):]
    return mask


# -----------------------------------------------------------------------------
# 核心：生成可投资掩码（逐日清洗，用于因子计算输入）
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
    from src.data.loader import load_field, to_monthly

    raw  = load_field(field, start=start, end=end)
    mask = build_investable_mask(start=start, end=end, freq="D")

    if freq == "M":
        raw  = to_monthly(raw)
        mask = mask.resample("ME").last().fillna(False)

    return apply_universe(raw, mask)
