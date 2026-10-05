# =============================================================================
# strategy/optimizer.py
# 组合构建与优化
#
# 基于综合因子得分构建投资组合，支持：
#   - 分组等权（Quintile / Decile，来自 因子框架.py 框架）
#   - 多空组合（Long Top - Short Bottom）
#   - 风险预算（TODO：后续扩展）
#
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional


def group_by_score(
    score: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """
    按因子得分横截面分组（每个时间截面独立分组）。

    Parameters
    ----------
    score : pd.DataFrame
        综合因子得分，index=日期，columns=股票代码
    n_groups : int
        分组数，默认 5（五分位），可选 10（十分位）

    Returns
    -------
    pd.DataFrame
        分组标签，值为 1（最低分组）到 n_groups（最高分组），NaN 表示无效
    """
    result = score.copy() * np.nan
    probs = np.linspace(0, 1, n_groups + 1)

    for date in score.index:
        row = score.loc[date].dropna()
        if len(row) < n_groups * 5:  # 每组至少 5 只股票
            continue
        # 成员划分与 pd.qcut 完全一致：同一分位实现（Series.quantile，线性插值）、右闭区间
        # （x ≤ 第 i 个分位点 → 第 i 组），取值相同的股票必落同一组。
        edges = row.quantile(probs).to_numpy()
        labels = np.searchsorted(edges[1:-1], row.to_numpy(), side="left") + 1
        # 标签锚定两端（待办 #36，2026-10-05）：取值大量并列时分位点重合、部分组为空，
        # 原 qcut(duplicates="drop", labels=False) 会把剩下的组**从 0 重新编号**，
        # 最高一档被标成 G4 / G3，G5 与多空（G5−G1）整月变 NaN（ni_inc8q 235/235 个月、
        # f_score 110 个月）。此处保留原分位编号，并令最低一档 = G1、最高一档 = G5。
        # 无并列的月份与原实现逐格相同（113 个因子中 108 个完全不变）。
        present = np.unique(labels)
        if present.size >= 2:
            labels = np.where(labels == present.min(), 1,
                              np.where(labels == present.max(), n_groups, labels))
        result.loc[date, row.index] = labels

    return result


def long_short_weights(
    groups: pd.DataFrame,
    n_groups: int = 5,
    long_group: Optional[int] = None,
    short_group: Optional[int] = None,
) -> pd.DataFrame:
    """
    构建多空等权权重矩阵。

    Parameters
    ----------
    groups : pd.DataFrame
        由 group_by_score() 输出的分组标签
    n_groups : int
        总分组数
    long_group : int, optional
        做多分组（默认最高分组 = n_groups）
    short_group : int, optional
        做空分组（默认最低分组 = 1）

    Returns
    -------
    pd.DataFrame
        权重矩阵：做多为正权重（等权），做空为负权重（等权），其余为 0
    """
    long_group  = long_group  or n_groups
    short_group = short_group or 1

    weights = pd.DataFrame(0.0, index=groups.index, columns=groups.columns)

    for date in groups.index:
        row = groups.loc[date]

        long_stocks  = row[row == long_group].index
        short_stocks = row[row == short_group].index

        if len(long_stocks) > 0:
            weights.loc[date, long_stocks] = 1.0 / len(long_stocks)
        if len(short_stocks) > 0:
            weights.loc[date, short_stocks] = -1.0 / len(short_stocks)

    return weights
