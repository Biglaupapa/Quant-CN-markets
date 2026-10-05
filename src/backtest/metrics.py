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
    if n == 0:
        return np.nan
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

    # ── 为什么是向量化而不是逐月循环 ────────────────────────────────────────
    # 原实现按 date 循环 234 次，每次做 4 次基于标签的 pandas 索引
    # （`.loc[date, common_cols]` + 三次 reindex/dropna），common_cols 有
    # 5867 个标签。scipy.spearmanr 本身在 5000 个元素上是毫秒级的，
    # 真正的开销全在这 936 次宽表标签索引上——实测每月约 90ms，
    # 62 个因子合计 **21.7 分钟，占整轮回测耗时的 95.4%**。
    #
    # 改法与本框架已有的四处向量化（见 CLAUDE.md「性能」一节）同源：
    # 循环外一次性对齐，然后按行做矩阵运算。
    #
    # 秩相关 = 秩上的 Pearson 相关，这是 Spearman 的定义本身，不是近似。
    # 关键是**掩码要先于排秩**：原实现是逐月把 factor 与 forward_ret 都非空
    # 的那批股票挑出来、再在这个存活子集上算秩。若先排秩后掩码，秩的分母
    # 会变，结果就不同了。下面 `where(mask)` 正是在复现这个顺序。
    F = factor.loc[common_idx, common_cols]
    R = forward_ret.loc[common_idx, common_cols]

    mask = F.notna() & R.notna()          # 逐格「两边都有值」
    F = F.where(mask)
    R = R.where(mask)

    if method == "spearman":
        # rank 默认 method="average"（并列取平均秩）、na_option="keep"，
        # 与 scipy.stats.rankdata 的默认行为一致。
        F = F.rank(axis=1)
        R = R.rank(axis=1)

    m = mask.to_numpy()
    n = m.sum(axis=1)
    a = np.nan_to_num(F.to_numpy(dtype=np.float64))   # 掩掉的格填 0
    b = np.nan_to_num(R.to_numpy(dtype=np.float64))

    # 按行去均值再算 Pearson。均值用掩码手算而不用 np.nanmean：
    # 整行全 NaN 时 nanmean 会抛 "Mean of empty slice" RuntimeWarning，
    # 而早期月份（2007 年前后上市公司少）这种行确实存在。
    # 去均值后必须再乘一次掩码——否则被掩掉的格会变成 −mean 混进求和。
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (a - (a.sum(axis=1) / n)[:, None]) * m
        b = (b - (b.sum(axis=1) / n)[:, None]) * m

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
        ic = np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)

    # 原实现的 `len(both) < 10 → NaN` 规则，原样保留
    ic = np.where(n < 10, np.nan, ic)

    return pd.Series(ic, index=common_idx)


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
