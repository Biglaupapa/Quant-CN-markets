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
from sklearn.linear_model import LinearRegression
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
    result = factor.copy() * np.nan
    reg    = LinearRegression()

    common_idx  = factor.index.intersection(log_mktcap.index)
    common_cols = factor.columns.intersection(log_mktcap.columns)

    for date in common_idx:
        y    = factor.loc[date, common_cols]
        x    = log_mktcap.loc[date, common_cols]
        mask = y.notna() & x.notna()

        if mask.sum() < MIN_ROLLING_VALID_DAYS:
            continue

        y_clean = y[mask].values.reshape(-1, 1)
        x_clean = x[mask].values.reshape(-1, 1)

        reg.fit(x_clean, y_clean)
        residuals = y_clean.flatten() - reg.predict(x_clean).flatten()

        result.loc[date, common_cols[mask]] = residuals

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
    reg    = LinearRegression()

    try:
        store = pd.HDFStore(str(industry_h5_path), mode="r")
        keys  = store.keys()
    except Exception as e:
        warnings.warn(f"[base] 无法打开行业 H5 文件: {e}")
        return factor

    for date in factor.index:
        date_str = date.strftime("%Y-%m-%d")
        y = factor.loc[date].dropna()

        if len(y) < MIN_ROLLING_VALID_DAYS:
            continue

        # 从 HDF5 中收集该日期的行业哑变量
        industry_dummies = []
        for key in keys:
            try:
                ind_df = store[key]
                if date_str in ind_df.index:
                    row = ind_df.loc[date_str]
                    industry_dummies.append(row.reindex(y.index).fillna(0))
            except Exception:
                continue

        if not industry_dummies:
            continue

        X = pd.DataFrame(industry_dummies).T  # shape: (n_stocks, n_industries)
        X = X.reindex(y.index).fillna(0)

        # 去掉全零列（该日期无该行业股票）
        X = X.loc[:, X.sum() > 0]
        if X.shape[1] == 0:
            continue

        try:
            reg.fit(X.values, y.values)
            residuals = y.values - reg.predict(X.values)
            result.loc[date, y.index] = residuals
        except Exception:
            continue

    store.close()
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
    reg    = LinearRegression()

    common_idx  = factor.index.intersection(log_mktcap.index)
    common_cols = factor.columns.intersection(log_mktcap.columns)

    try:
        store = pd.HDFStore(str(industry_h5_path), mode="r")
        keys  = store.keys()
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

        # 行业哑变量
        industry_dummies = []
        for key in keys:
            try:
                ind_df = store[key]
                if date_str in ind_df.index:
                    row = ind_df.loc[date_str]
                    industry_dummies.append(row.reindex(y_s.index).fillna(0))
            except Exception:
                continue

        # 构建 X 矩阵：[log_mktcap | industry_dummies]
        X = pd.DataFrame({"log_mktcap": size_s})
        if industry_dummies:
            ind_df_cross = pd.DataFrame(industry_dummies).T.reindex(y_s.index).fillna(0)
            ind_df_cross = ind_df_cross.loc[:, ind_df_cross.sum() > 0]
            X = pd.concat([X, ind_df_cross], axis=1)

        try:
            reg.fit(X.values, y_s.values)
            residuals = y_s.values - reg.predict(X.values)
            result.loc[date, y_s.index] = residuals
        except Exception:
            continue

    store.close()
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
