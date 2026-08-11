# =============================================================================
# factors/base.py
# 因子基类：提供标准化、去极值、中性化三个通用方法
#
# 所有因子（microstructure.py / fundamental.py）均调用本模块的工具函数。
# 不需要实例化，直接调用静态函数即可。
#
# 中性化说明（经用户确认）：
#   - 市值中性化：仅 Turnover 因子使用
#   - 市值+行业双重中性化：PB、NetProfit_YoY 使用
#   - 其他因子：暂不做中性化
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
# 2026-08 起三个中性化函数改用 np.linalg.lstsq（与 sklearn 同为 SVD 最小二乘，
# 数学等价、无对象创建开销），不再依赖 sklearn。
from typing import Optional, List
import warnings

from src.config.settings import INDUSTRY_H5_PATH, MIN_ROLLING_VALID_DAYS


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


def neutralize_by_industry(
    factor: pd.DataFrame,
    industry_h5_path: Path = INDUSTRY_H5_PATH,
) -> pd.DataFrame:
    """
    行业中性化：截面回归剔除因子与行业虚拟变量的线性相关部分，取残差。

    使用 FactorLoading_Industry_arch.h5 中的行业分类数据。
    经用户确认：PB 和 NetProfit_YoY 做市值+行业双重中性化。

    Parameters
    ----------
    factor : pd.DataFrame
        因子值，index=日期（月度），columns=股票代码
    industry_h5_path : Path
        行业因子 HDF5 文件路径

    Returns
    -------
    pd.DataFrame
        行业中性化后的因子值（残差）
    """
    if not industry_h5_path.exists():
        warnings.warn(f"[base] 行业 H5 文件不存在: {industry_h5_path}，跳过行业中性化。")
        return factor

    result = factor.copy() * np.nan

    # ── 优化说明 ──────────────────────────────────────────────────
    # 原实现把 `store[key]` 放在「日期 × 行业」双重循环内部，
    # 每个日期都把整个 HDF5 的所有行业表重读一遍——IO 是主要开销。
    # 现改为循环外一次性载入内存；回归用 lstsq 取代 sklearn。
    # 数学等价（同为最小二乘），差异仅浮点累加顺序。
    try:
        with pd.HDFStore(str(industry_h5_path), mode="r") as store:
            industry_tables = {k: store[k] for k in store.keys()}
    except Exception as e:
        warnings.warn(f"[base] 无法打开行业 H5 文件: {e}")
        return factor

    if not industry_tables:
        return result

    for date in factor.index:
        date_str = date.strftime("%Y-%m-%d")
        y = factor.loc[date].dropna()

        if len(y) < MIN_ROLLING_VALID_DAYS:
            continue

        # 从内存中收集该日期的行业哑变量
        industry_dummies = [
            tbl.loc[date_str].reindex(y.index).fillna(0)
            for tbl in industry_tables.values()
            if date_str in tbl.index
        ]
        if not industry_dummies:
            continue

        X = pd.DataFrame(industry_dummies).T          # (n_stocks, n_industries)
        X = X.reindex(y.index).fillna(0)

        # 去掉全零列（该日期无该行业股票）
        X = X.loc[:, X.sum() > 0]
        if X.shape[1] == 0:
            continue

        # sklearn 默认 fit_intercept=True，等价于在设计矩阵中加一列常数
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
    industry_h5_path: Path = INDUSTRY_H5_PATH,
) -> pd.DataFrame:
    """
    市值 + 行业双重中性化：截面同时对 log(市值) 和行业哑变量做回归，取残差。

    经用户确认：PB 和 NetProfit_YoY 使用此方法。

    Parameters
    ----------
    factor : pd.DataFrame
        因子值，index=日期（月度），columns=股票代码
    log_mktcap : pd.DataFrame
        log(流通市值)
    industry_h5_path : Path
        行业因子 HDF5 文件路径

    Returns
    -------
    pd.DataFrame
        双重中性化后的因子值（残差）
    """
    if not industry_h5_path.exists():
        warnings.warn(f"[base] 行业 H5 文件不存在，退化为仅市值中性化。")
        return neutralize_by_size(factor, log_mktcap)

    result = factor.copy() * np.nan

    common_idx  = factor.index.intersection(log_mktcap.index)
    common_cols = factor.columns.intersection(log_mktcap.columns)

    # 同 neutralize_by_industry：HDF5 一次性载入内存，避免在双重循环里反复读盘
    try:
        with pd.HDFStore(str(industry_h5_path), mode="r") as store:
            industry_tables = {k: store[k] for k in store.keys()}
    except Exception as e:
        warnings.warn(f"[base] 无法打开行业 H5 文件: {e}，退化为仅市值中性化。")
        return neutralize_by_size(factor, log_mktcap)

    for date in common_idx:
        date_str = date.strftime("%Y-%m-%d")
        y    = factor.loc[date, common_cols]
        size = log_mktcap.loc[date, common_cols]
        mask = y.notna() & size.notna()

        if mask.sum() < MIN_ROLLING_VALID_DAYS:
            continue

        y_s    = y[mask]
        size_s = size[mask]

        # 行业哑变量（从内存取）
        industry_dummies = [
            tbl.loc[date_str].reindex(y_s.index).fillna(0)
            for tbl in industry_tables.values()
            if date_str in tbl.index
        ]

        # 构建 X 矩阵：[log_mktcap | industry_dummies]
        X = pd.DataFrame({"log_mktcap": size_s})
        if industry_dummies:
            ind_df_cross = pd.DataFrame(industry_dummies).T.reindex(y_s.index).fillna(0)
            ind_df_cross = ind_df_cross.loc[:, ind_df_cross.sum() > 0]
            X = pd.concat([X, ind_df_cross], axis=1)

        # sklearn 默认 fit_intercept=True → 设计矩阵补一列常数
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
    industry_h5_path: Path = INDUSTRY_H5_PATH,
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
    industry_h5_path : Path
        行业中性化 HDF5 路径

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
        f = neutralize_by_industry(f, industry_h5_path)

    elif neutralize == "size+industry":
        if log_mktcap is None:
            raise ValueError("[base] 市值+行业中性化需提供 log_mktcap。")
        f = neutralize_by_size_and_industry(f, log_mktcap, industry_h5_path)

    elif neutralize is not None:
        raise ValueError(f"[base] 不支持的中性化方式: {neutralize}")

    # Step 3: 标准化
    if do_standardize:
        f = standardize(f)

    return f
