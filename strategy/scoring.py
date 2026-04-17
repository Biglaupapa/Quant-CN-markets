# =============================================================================
# strategy/scoring.py
# 多因子打分与加权
#
# 将多个因子合成为一个综合打分，支持：
#   - 等权合成（Equal Weight）
#   - IC 加权合成（历史 IC 均值加权）
#   - ICIR 加权合成（IC 均值/IC 标准差加权，稳定性更优）
#
# TODO: 后续可扩展机器学习加权（XGBoost / Ridge 回归）
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Optional


def equal_weight(factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    等权合成：对所有因子的标准化值取简单平均。

    Parameters
    ----------
    factors : dict
        {factor_name: pd.DataFrame}，每个 DataFrame 应已标准化（Z-score）

    Returns
    -------
    pd.DataFrame
        综合得分，index=日期，columns=股票代码
    """
    if not factors:
        raise ValueError("[scoring] factors 为空")

    all_dfs = list(factors.values())
    # 对齐所有因子的 index 和 columns
    common_idx  = all_dfs[0].index
    common_cols = all_dfs[0].columns
    for df in all_dfs[1:]:
        common_idx  = common_idx.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)

    aligned = [df.reindex(index=common_idx, columns=common_cols) for df in all_dfs]
    score   = pd.concat(aligned, axis=0).groupby(level=0).mean()
    # 注意：concat + groupby 是简便写法，等效于 stack → mean
    score = sum(df.reindex(index=common_idx, columns=common_cols)
                for df in aligned) / len(aligned)
    return score


def ic_weighted(
    factors: Dict[str, pd.DataFrame],
    forward_ret: pd.DataFrame,
    ic_window: int = 12,
    method: str = "icir",
) -> pd.DataFrame:
    """
    IC / ICIR 加权合成。

    先计算每个因子的滚动历史 IC（Rank IC），
    以 IC 均值（或 IC均值/IC标准差）作为权重合成综合打分。

    Parameters
    ----------
    factors : dict
        {factor_name: pd.DataFrame}，已标准化
    forward_ret : pd.DataFrame
        下期收益率，用于计算 IC
    ic_window : int
        计算历史 IC 均值的滚动窗口（月）
    method : str
        "ic"   → IC 均值加权
        "icir" → ICIR（IC均值/IC标准差）加权

    Returns
    -------
    pd.DataFrame
        IC 加权综合得分
    """
    # TODO: 实现 IC 加权合成
    raise NotImplementedError("[scoring] IC 加权尚未实现，请先使用 equal_weight()")
