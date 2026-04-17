# =============================================================================
# backtest/metrics.py
# 绩效指标计算
#
# 包含：
#   - 分组收益统计（年化收益、波动率、夏普比率、最大回撤、胜率）
#   - IC / RankIC 分析（月度截面相关性、ICIR）
#   - 多空对冲统计
#
# 来源：[orig] 因子框架.py 绩效指标 + [af] fama_macbeth.py IC 分析
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


# -----------------------------------------------------------------------------
# 基础统计指标
# -----------------------------------------------------------------------------

def annualized_return(ret_series: pd.Series, freq: int = 12) -> float:
    """年化收益率（复利）"""
    n = len(ret_series.dropna())
    cumret = (1 + ret_series.dropna()).prod()
    return cumret ** (freq / n) - 1


def annualized_volatility(ret_series: pd.Series, freq: int = 12) -> float:
    """年化波动率"""
    return ret_series.dropna().std() * np.sqrt(freq)


def sharpe_ratio(ret_series: pd.Series, freq: int = 12, rf: float = 0.0) -> float:
    """信息比率（Sharpe，无风险利率默认为 0）"""
    ann_ret = annualized_return(ret_series, freq) - rf
    ann_vol = annualized_volatility(ret_series, freq)
    return ann_ret / ann_vol if ann_vol > 0 else np.nan


def max_drawdown(ret_series: pd.Series) -> float:
    """最大回撤"""
    cumret = (1 + ret_series.fillna(0)).cumprod()
    roll_max = cumret.cummax()
    drawdown = (cumret - roll_max) / roll_max
    return drawdown.min()


def win_rate(ret_series: pd.Series) -> float:
    """月度胜率（正收益月份占比）"""
    valid = ret_series.dropna()
    return (valid > 0).sum() / len(valid) if len(valid) > 0 else np.nan


def group_summary(group_ret: pd.DataFrame, freq: int = 12) -> pd.DataFrame:
    """
    对分组收益表格生成汇总统计。

    Parameters
    ----------
    group_ret : pd.DataFrame
        由 engine.group_return() 输出，columns 包含 G1..Gn 和 LS
    freq : int
        数据频率（月度=12，季度=4）

    Returns
    -------
    pd.DataFrame
        汇总指标表，index=组别名称，columns=["年化收益","年化波动","夏普","最大回撤","胜率"]
    """
    records = {}
    for col in group_ret.columns:
        s = group_ret[col]
        records[col] = {
            "年化收益":  annualized_return(s, freq),
            "年化波动":  annualized_volatility(s, freq),
            "夏普比率":  sharpe_ratio(s, freq),
            "最大回撤":  max_drawdown(s),
            "月度胜率":  win_rate(s),
        }
    return pd.DataFrame(records).T


# -----------------------------------------------------------------------------
# IC / RankIC 分析
# -----------------------------------------------------------------------------

def calc_ic(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    计算月度截面 IC（Information Coefficient）。

    Parameters
    ----------
    factor : pd.DataFrame
        因子值，index=月末日期，columns=股票代码
    forward_ret : pd.DataFrame
        下期收益率（factor 的 shift(-1)），相同维度
    method : str
        "pearson"  → Pearson IC（线性相关）
        "spearman" → Rank IC（秩相关，更稳健，默认）

    Returns
    -------
    pd.Series
        月度 IC 序列
    """
    common_idx  = factor.index.intersection(forward_ret.index)
    common_cols = factor.columns.intersection(forward_ret.columns)

    ic_series = {}
    for date in common_idx:
        f = factor.loc[date, common_cols].dropna()
        r = forward_ret.loc[date, common_cols].reindex(f.index).dropna()
        both = f.reindex(r.index).dropna()
        r    = r.reindex(both.index)

        if len(both) < 10:
            ic_series[date] = np.nan
            continue

        if method == "spearman":
            ic, _ = stats.spearmanr(both.values, r.values)
        else:
            ic, _ = stats.pearsonr(both.values, r.values)

        ic_series[date] = ic

    return pd.Series(ic_series)


def ic_summary(ic_series: pd.Series) -> dict:
    """
    IC 统计汇总：均值、标准差、ICIR、显著月份占比。

    Returns
    -------
    dict
        {"IC均值", "IC标准差", "ICIR", "|IC|>0.02占比", "IC>0占比"}
    """
    valid = ic_series.dropna()
    return {
        "IC均值":       valid.mean(),
        "IC标准差":     valid.std(),
        "ICIR":         valid.mean() / valid.std() if valid.std() > 0 else np.nan,
        "|IC|>0.02占比": (valid.abs() > 0.02).mean(),
        "IC>0占比":     (valid > 0).mean(),
    }
