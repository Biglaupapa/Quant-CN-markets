# =============================================================================
# backtest/engine.py
# 月度/季度分组回测核心
#
# 回测逻辑（延续 因子框架.py 框架）：
#   - 月度换仓：月初开盘价买入，月末收盘价卖出
#   - 月度收益率 = 月末后复权收盘价 / 月初后复权开盘价 - 1
#   - 特殊处理：
#       * 涨停开盘（开盘价=涨停价）→ 无法买入，排除
#       * 跌停收盘（收盘价=跌停价）→ 无法卖出，特殊处理
#
# 来源：[orig] 因子框架.py 回测逻辑
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional

from src.data.loader import load_data, load_data_hk, to_monthly
from src.data.universe import build_investable_mask, build_investable_mask_hk, apply_universe


def calc_monthly_returns(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
) -> pd.DataFrame:
    """
    计算月度持仓收益率矩阵。

    月度收益率 = 月末后复权收盘价 / 月初后复权开盘价 - 1

    A 股特殊处理：
    - 涨停开盘（开盘 >= 前收 × 1.099）→ 置 NaN，无法建仓
    - 港股无每日涨跌停限制，跳过此过滤

    Parameters
    ----------
    market : str
        "A"（默认，A股）或 "HK"（港股）

    Returns
    -------
    pd.DataFrame
        月度收益率，index=月末日期，columns=股票代码
    """
    if market == "HK":
        data = load_data_hk(["close_adj", "open_adj"], start=start, end=end)
        mask = build_investable_mask_hk(start=start, end=end, freq="D")
    else:
        data = load_data(["close_adj", "open_adj"], start=start, end=end)
        mask = build_investable_mask(start=start, end=end, freq="D")

    close = data.get("close_adj")
    open_ = data.get("open_adj")

    if close is None or open_ is None:
        raise ValueError(f"[engine] 缺少 close_adj 或 open_adj 数据（market={market}）")

    close = apply_universe(close, mask)
    open_ = apply_universe(open_, mask)

    # 月初开盘价（月第一个交易日开盘）
    # replace(0, nan)：开盘价为零时（涨跌停锁板等异常）不能作分母，否则收益率为 inf
    open_month_start = open_.resample("ME").first().replace(0, np.nan)
    # 月末收盘价
    close_month_end  = close.resample("ME").last()

    # 月度收益
    monthly_ret = close_month_end / open_month_start - 1

    # A 股涨停开盘过滤：月初开盘价 >= 前月末收盘 × 1.099 → NaN（无法买入）
    # 港股无每日涨跌停限制，跳过此过滤
    if market != "HK":
        prev_close_m  = close_month_end.shift(1)
        limit_up_open = (open_month_start >= prev_close_m * 1.099)
        monthly_ret[limit_up_open] = np.nan

    return monthly_ret


def group_return(
    factor: pd.DataFrame,
    monthly_ret: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """
    分组回测：按因子得分分成 n_groups 组，计算每组等权月度收益率。

    Parameters
    ----------
    factor : pd.DataFrame
        月度因子值，index=月末日期，columns=股票代码
    monthly_ret : pd.DataFrame
        月度收益率（由 calc_monthly_returns() 计算）
    n_groups : int
        分组数（5 或 10）

    Returns
    -------
    pd.DataFrame
        各分组月度收益率，index=日期，columns=["G1","G2",...,"Gn","LS"]
        "LS" 列为多空组合（Gn - G1）
    """
    from strategy.optimizer import group_by_score

    # 对齐时间和股票
    common_idx  = factor.index.intersection(monthly_ret.index)
    common_cols = factor.columns.intersection(monthly_ret.columns)

    factor_aligned = factor.reindex(index=common_idx, columns=common_cols)
    ret_aligned    = monthly_ret.reindex(index=common_idx, columns=common_cols)

    groups = group_by_score(factor_aligned, n_groups=n_groups)

    records = []
    for date in common_idx:
        row_g   = groups.loc[date]
        row_ret = ret_aligned.loc[date]
        record  = {}
        for g in range(1, n_groups + 1):
            stocks_in_group = row_g[row_g == g].index
            valid_ret       = row_ret.reindex(stocks_in_group).dropna()
            record[f"G{g}"] = valid_ret.mean() if len(valid_ret) > 0 else np.nan
        record["LS"] = record.get(f"G{n_groups}", np.nan) - record.get("G1", np.nan)
        records.append(record)

    return pd.DataFrame(records, index=common_idx)
