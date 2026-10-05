# =============================================================================
# backtest/engine.py
# 月度/季度分组回测核心
#
# 回测逻辑（2026-10-01 起与文献对齐，见 docs/【方法】回测口径与文献对齐.md）：
#   - 月度换仓：t 月末组建（只用 t 月及以前信息筛选），持有 t+1 月
#   - 月度收益率 = t+1 月末后复权收盘价 / t 月末后复权收盘价 - 1（收盘到收盘，LSY / HQZ / JKP）
#   - 持有期不再筛选：t+1 月被戴 ST、停牌，收益照算（停牌日收盘价为停牌前价格，自然按最后价）
#   - 退市：收益算到最后交易日，不打折
#
# 2026-10-01 之前：月初开盘买入、月末收盘卖出，且持有期价格套用了 t+1 月逐日投资域掩码
# → 持有期被 ST / 停牌的股票收益被截断或丢弃（丢弃者真实收益中位 −6.3%），系统性高估收益。
# 开盘入场口径未找到文献采用，已删除。
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional

from src.data.loader import load_data, load_data_hk
from src.data.universe import build_formation_mask, build_investable_mask_hk


def formation_pool(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
    formation_config: Optional[dict] = None,
) -> pd.DataFrame:
    """t 月末组建日股票池（月末 × 股票，bool）。A 股 = build_formation_mask，港股 = 月度投资域。"""
    if market == "HK":
        return build_investable_mask_hk(start=start, end=end, freq="M")
    return build_formation_mask(start=start, end=end, config=formation_config)


def calc_monthly_returns(
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
    formation_config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    月度持有收益率矩阵（收盘到收盘，持有期不筛选）。

        ret[m] = close_adj[m 月最后交易日] / close_adj[m-1 月最后交易日] - 1

    index = 收益实现月的月末（与原约定一致：调用方用 `.shift(-1)` 得到 t 月因子对应的 t+1 月收益）。

    两层，各管一件事：

    1. **组建（t 月末）**：t 月末不在组建日股票池（universe.build_formation_mask）的股票，
       其 t+1 月收益置 NaN。只用 t 月末信息，
       在这里落实是因为所有下游（单因子 IC、分组、合成、ML 基准）都用这张矩阵——一处生效。
       （因子侧按日计算后取「当月最后有效值」，月末当天停牌 / ST 的股票仍带因子值，
       2026-10-01 实测占各因子 1%~3%；不能依赖因子侧自己剔除。）
    2. **持有（t+1 月）**：价格**不套投资域掩码**。入组后发生的 ST、停牌、退市都按真实价格计入
       ——LSY / HQZ / JKP 的口径，避免「持续筛选预先排除未来输家」。

    - 停牌：原始数据停牌日 CLOSE 为停牌前价格 → 自然按最后价，整月停牌收益为 0
    - 退市：月内最后交易日之后无价格；若当月一天未交易就退市，`ffill(limit=1)` 让该月
      收益为 0（按最后价退出），再往后不延续
    - 不打退市折价（A 股文献未加；退市整理期跌幅已在价格中）

    Parameters
    ----------
    market : str
        "A"（默认，A股）或 "HK"（港股）
    formation_config : dict, optional
        覆盖 settings.FORMATION_CONFIG（仅 A 股），用于逐条规则对比
    """
    loader = load_data_hk if market == "HK" else load_data
    close = loader(["close_adj"], start=start, end=end).get("close_adj")
    if close is None:
        raise ValueError(f"[engine] 缺少 close_adj 数据（market={market}）")

    # 持有：不套掩码的月末（或当月最后可得）收盘价
    close_me = close.resample("ME").last().ffill(limit=1)
    ret = close_me / close_me.shift(1) - 1

    # 组建：t 月末可投资 → 才有 t+1 月收益（shift(1) 把 t 月末的判断对齐到 t+1 行）
    formed = formation_pool(start, end, market, formation_config).shift(1)
    formed = formed.reindex(index=ret.index, columns=ret.columns).fillna(False).astype(bool)
    return ret.where(formed)


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
    from src.strategy.optimizer import group_by_score

    # 对齐时间和股票
    common_idx  = factor.index.intersection(monthly_ret.index)
    common_cols = factor.columns.intersection(monthly_ret.columns)

    factor_aligned = factor.reindex(index=common_idx, columns=common_cols)
    ret_aligned    = monthly_ret.reindex(index=common_idx, columns=common_cols)

    # 分位点只在组建日股票池内切（与 calc_ic 先掩码后排秩一致）。
    # calc_monthly_returns 对不在 t 月末股票池的股票置 NaN，故「下期收益非空」即入池。
    # 原实现在因子全截面上切分位、求均值时才丢弃无收益股票：规则 g 打开时最小 30%
    # 有因子值无收益，size 的 G1 有 196/237 个月整组为空（2026-10-04 发现）。
    factor_aligned = factor_aligned.where(ret_aligned.notna())

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
