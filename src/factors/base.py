# =============================================================================
# factors/base.py
# 因子基类：提供标准化、去极值、中性化三个通用方法
#
# 所有因子（microstructure.py / fundamental.py）均调用本模块的工具函数。
# 不需要实例化，直接调用静态函数即可。
#
# 中性化说明（经用户确认）：
#   - 市值中性化：仅 Turnover 因子使用
#   - 市值+行业双重中性化：会计因子（accounting.py）、NetProfit_YoY 使用；行业 = 聚源申万一级（时点）
#   - 其他因子：暂不做中性化
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
# 2026-08 起三个中性化函数改用 np.linalg.lstsq（与 sklearn 同为 SVD 最小二乘，
# 数学等价、无对象创建开销），不再依赖 sklearn。
from typing import Optional, List
import warnings

from src.config.settings import INDUSTRY_PATH, MIN_ROLLING_VALID_DAYS


# -----------------------------------------------------------------------------
# 1. 去极值（Winsorization）
# -----------------------------------------------------------------------------

def winsorize(
    factor: pd.DataFrame,
    n_std: float = 3.0,
    method: str = "clip",
) -> pd.DataFrame:
    """
    横截面去极值：对每个截面（每行/每个时间点）独立处理。

    Parameters
    ----------
    factor : pd.DataFrame
        因子值，index=日期，columns=股票代码
    n_std : float
        极值判断的标准差倍数，默认 3.0
    method : str
        "clip"  → 将极值截断至边界（保留样本数）
        "remove"→ 将极值置为 NaN（减少样本数）

    Returns
    -------
    pd.DataFrame
        去极值后的因子值，形状与输入相同
    """
    result = factor.copy()

    for date in result.index:
        row = result.loc[date].dropna()
        if len(row) < 5:
            continue
        mu  = row.mean()
        std = row.std()
        lo  = mu - n_std * std
        hi  = mu + n_std * std

        if method == "clip":
            result.loc[date] = result.loc[date].clip(lower=lo, upper=hi)
        elif method == "remove":
            mask = (result.loc[date] < lo) | (result.loc[date] > hi)
            result.loc[date, mask] = np.nan
        else:
            raise ValueError(f"[base] 不支持的方法: {method}")

    return result


# -----------------------------------------------------------------------------
# 2. 标准化（Z-score，横截面）
# -----------------------------------------------------------------------------

def standardize(factor: pd.DataFrame) -> pd.DataFrame:
    """
    横截面标准化：每个时间截面独立做 Z-score。
    Z = (X - mean) / std

    Parameters
    ----------
    factor : pd.DataFrame
        因子值，index=日期，columns=股票代码

    Returns
    -------
    pd.DataFrame
        标准化后的因子值
    """
    mean = factor.mean(axis=1)
    std  = factor.std(axis=1)
    # 避免标准差为 0 时除法报错
    std  = std.replace(0, np.nan)
    result = factor.sub(mean, axis=0).div(std, axis=0)
    return result


# -----------------------------------------------------------------------------
# 3. 中性化（Neutralization）
# -----------------------------------------------------------------------------

def neutralize_by_size(
    factor: pd.DataFrame,
    log_mktcap: pd.DataFrame,
) -> pd.DataFrame:
    """
    市值中性化：截面回归剔除因子与 log(流通市值) 的线性相关部分，取残差。

    经用户确认：仅 Turnover 因子使用此方法。

    Parameters
    ----------
    factor : pd.DataFrame
        因子值，index=日期，columns=股票代码
    log_mktcap : pd.DataFrame
        log(流通市值)，与 factor 维度一致

    Returns
    -------
    pd.DataFrame
        市值中性化后的因子值（残差）
    """
    common_idx  = factor.index.intersection(log_mktcap.index)
    common_cols = factor.columns.intersection(log_mktcap.columns)

    Y = factor.loc[common_idx, common_cols]
    X = log_mktcap.loc[common_idx, common_cols]

    # ── 向量化说明 ────────────────────────────────────────────────
    # 原实现是逐日期 `LinearRegression().fit()`（260 期 × 每个中性化因子）。
    # 单变量 OLS 有闭式解，无需迭代求解器：
    #     β = Cov(x, y) / Var(x)      α = ȳ - β·x̄
    #     resid = y - (α + β·x) = (y - ȳ) - β·(x - x̄)
    # 与 sklearn 数学等价（sklearn 内部同样是最小二乘正规方程），
    # 差异仅在浮点累加顺序，量级 ~1e-15。
    #
    # 两侧的均值都只在「x、y 同时非空」的截面上计算，与原来的
    # mask = y.notna() & x.notna() 完全一致。
    mask = Y.notna() & X.notna()
    Ym = Y.where(mask)
    Xm = X.where(mask)

    n = mask.sum(axis=1)
    ybar = Ym.mean(axis=1)
    xbar = Xm.mean(axis=1)

    dy = Ym.sub(ybar, axis=0)
    dx = Xm.sub(xbar, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (dx * dy).sum(axis=1) / (dx ** 2).sum(axis=1)

    resid = dy.sub(dx.mul(beta, axis=0))

    # 有效样本不足的日期整行置 NaN（原实现是 continue，即保持全 NaN）
    resid[n < MIN_ROLLING_VALID_DAYS] = np.nan

    # 输出对齐回原始形状：只在 mask 位置有值，其余 NaN
    result = factor.copy() * np.nan
    result.loc[common_idx, common_cols] = resid.where(mask)
    return result


def _load_industry(industry_path: Path, dates: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """
    读行业宽表并对齐到因子日期。

    行业宽表：行 = 日历月末，列 = 股票代码，值 = 该月最后交易日生效的申万一级行业代码
    （聚源「申万行业分类(新)」，时点口径；见 Database/docs/【登记】指标权威来源.md）。
    因子日期若不是月末，取不晚于该日的最近一个月末（ffill），不前视。

    2026-10-02 替代 Wind h5：h5 一直未生效——环境缺 pytables 打不开；即使打开，
    其索引为 'YYYYMMDD' 交易日，而旧代码按 'YYYY-MM-DD' 日历月末查找，永远匹配不上。
    """
    if not industry_path.exists():
        warnings.warn(f"[base] 行业文件不存在: {industry_path}")
        return None
    ind = pd.read_csv(industry_path, index_col=0)
    ind.index = pd.to_datetime(ind.index)
    ind = ind.sort_index()
    # 只按「行」对齐到不晚于因子日期的最近月末；**不**对单元格做 ffill——
    # 某月无行业的股票保持 NaN，不能把它以前（甚至退市前）的行业代码沿用下去
    return ind.reindex(pd.DatetimeIndex(dates), method="ffill")


def _industry_dummies(codes: pd.Series) -> pd.DataFrame:
    """一个截面的行业代码 → 哑变量（无行业的股票全 0）。"""
    return pd.get_dummies(codes.dropna().astype("int64").astype(str)).reindex(codes.index).fillna(0).astype(float)


def neutralize_by_industry(
    factor: pd.DataFrame,
    industry_path: Path = INDUSTRY_PATH,
) -> pd.DataFrame:
    """
    行业中性化：截面回归剔除因子与行业虚拟变量的线性相关部分，取残差。
    行业 = 聚源申万一级（时点），见 _load_industry。
    """
    ind = _load_industry(industry_path, factor.index)
    if ind is None:
        warnings.warn("[base] 无行业数据，跳过行业中性化。")
        return factor

    result = factor.copy() * np.nan
    for date in factor.index:
        y = factor.loc[date].dropna()
        if len(y) < MIN_ROLLING_VALID_DAYS:
            continue
        X = _industry_dummies(ind.loc[date].reindex(y.index))
        X = X.loc[:, X.sum() > 0]
        if X.shape[1] == 0:
            continue
        Xd = np.column_stack([np.ones(len(y)), X.values])
        try:
            beta, *_ = np.linalg.lstsq(Xd, y.values, rcond=None)
            result.loc[date, y.index] = y.values - Xd @ beta
        except Exception:
            continue
    return result


def neutralize_by_size_and_industry(
    factor: pd.DataFrame,
    log_mktcap: pd.DataFrame,
    industry_path: Path = INDUSTRY_PATH,
) -> pd.DataFrame:
    """
    市值 + 行业双重中性化：截面同时对 log(市值) 和行业哑变量做回归，取残差。
    行业 = 聚源申万一级（时点），见 _load_industry。无行业数据的股票行业哑变量全 0
    （仍参与回归，只由截距与市值解释）。
    """
    ind = _load_industry(industry_path, factor.index)
    if ind is None:
        warnings.warn("[base] 无行业数据，退化为仅市值中性化。")
        return neutralize_by_size(factor, log_mktcap)

    result = factor.copy() * np.nan
    common_idx  = factor.index.intersection(log_mktcap.index)
    common_cols = factor.columns.intersection(log_mktcap.columns)

    for date in common_idx:
        y    = factor.loc[date, common_cols]
        size = log_mktcap.loc[date, common_cols]
        mask = y.notna() & size.notna()
        if mask.sum() < MIN_ROLLING_VALID_DAYS:
            continue
        y_s, size_s = y[mask], size[mask]

        X = pd.DataFrame({"log_mktcap": size_s})
        D = _industry_dummies(ind.loc[date].reindex(y_s.index))
        # 保留全部行业哑变量：截距 + 全部哑变量共线时 lstsq 取最小范数解，残差（投影）不变；
        # 若删掉一个「基准行业」，当月无行业的股票会和该行业并成一组，两组均值都不再为 0
        D = D.loc[:, D.sum() > 0]
        X = pd.concat([X, D], axis=1)

        Xd = np.column_stack([np.ones(len(y_s)), X.values])
        try:
            beta, *_ = np.linalg.lstsq(Xd, y_s.values, rcond=None)
            result.loc[date, y_s.index] = y_s.values - Xd @ beta
        except Exception:
            continue

    return result


# -----------------------------------------------------------------------------
# 4. 完整因子预处理流水线
# -----------------------------------------------------------------------------

def preprocess(
    factor: pd.DataFrame,
    winsorize_n_std: float = 3.0,
    winsorize_method: str = "clip",
    do_standardize: bool = True,
    neutralize: Optional[str] = None,   # None | "size" | "industry" | "size+industry"
    log_mktcap: Optional[pd.DataFrame] = None,
    industry_path: Path = INDUSTRY_PATH,
) -> pd.DataFrame:
    """
    完整因子预处理：去极值 → 中性化 → 标准化（按顺序执行）。

    Parameters
    ----------
    factor : pd.DataFrame
        原始因子值
    winsorize_n_std : float
        去极值标准差倍数
    winsorize_method : str
        "clip" 或 "remove"
    do_standardize : bool
        是否在最后做截面标准化
    neutralize : str or None
        中性化方式，None 表示不做
    log_mktcap : pd.DataFrame, optional
        市值中性化所需的 log(流通市值)
    industry_path : Path
        行业宽表路径（聚源申万一级，时点）

    Returns
    -------
    pd.DataFrame
        预处理完成的因子值
    """
    # Step 1: 去极值
    f = winsorize(factor, n_std=winsorize_n_std, method=winsorize_method)

    # Step 2: 中性化
    if neutralize == "size":
        if log_mktcap is None:
            raise ValueError("[base] 市值中性化需提供 log_mktcap。")
        f = neutralize_by_size(f, log_mktcap)

    elif neutralize == "industry":
        f = neutralize_by_industry(f, industry_path)

    elif neutralize == "size+industry":
        if log_mktcap is None:
            raise ValueError("[base] 市值+行业中性化需提供 log_mktcap。")
        f = neutralize_by_size_and_industry(f, log_mktcap, industry_path)

    elif neutralize is not None:
        raise ValueError(f"[base] 不支持的中性化方式: {neutralize}")

    # Step 3: 标准化
    if do_standardize:
        f = standardize(f)

    return f
