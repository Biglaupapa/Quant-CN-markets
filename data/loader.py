# =============================================================================
# data/loader.py
# 统一数据加载接口
#
# 负责将两个来源的数据拼接成统一格式：
#   - 历史存档 CSV（2014-2020）：原始格式为 index=股票代码, columns=日期（转置）
#   - Database CSV（2021-至今）：标准格式为 index=日期, columns=股票代码
#
# 输出统一格式：index=日期(DatetimeIndex), columns=股票代码
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

from config.settings import (
    ARCHIVE_DATA_DIR,
    DATABASE_DIR,
    ARCHIVE_FIELD_MAP,
    DATABASE_FIELD_MAP,
    HIST_START,
    HIST_END,
    DB_START,
)


# -----------------------------------------------------------------------------
# 内部工具函数
# -----------------------------------------------------------------------------

def _read_archive_csv(filename: str, numeric: bool = True) -> pd.DataFrame:
    """
    读取历史存档 CSV。
    原始格式：index=股票代码, columns=日期字符串（转置格式）
    输出格式：index=DatetimeIndex(日频), columns=股票代码

    Parameters
    ----------
    numeric : bool
        True（默认）：将数据列转换为 float（适用于价格、换手率等数值字段）
        False：保留原始字符串（适用于 trade_status 等文字字段）
    """
    path = ARCHIVE_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"[loader] 存档文件不存在: {path}")

    df = pd.read_csv(path, index_col=0, low_memory=False)
    # 转置：行变列（日期变为 index，股票代码变为 columns）
    df = df.T
    # 存档 CSV 转置后 index 可能混入非日期行，统一清理：
    #   - "日期"（原始列名 artifact）
    #   - 尾部空格（如 "20140103 "）
    #   - 空白行
    #   - "Unnamed: N"（CSV 空列转置后的残留）
    # 只保留严格的 8 位纯数字字符串（YYYYMMDD）
    df.index = df.index.str.strip()
    df = df[df.index.str.match(r"^\d{8}$")]
    # 存档日期格式固定为 YYYYMMDD，显式指定避免 pandas 误推断
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.sort_index()
    # 裁剪至存档时间范围
    df = df.loc[HIST_START:HIST_END]
    # 数值转换（trade_status 等字符串字段不做转换）
    if numeric:
        df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _read_database_csv(filename: str) -> pd.DataFrame:
    """
    读取 Database CSV。
    标准格式：index=日期字符串, columns=股票代码
    输出格式：index=DatetimeIndex(日频), columns=股票代码
    """
    path = DATABASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"[loader] Database 文件不存在: {path}")

    df = pd.read_csv(path, index_col=0, low_memory=False)
    # Database 日期格式可能为 YYYY-MM-DD 或 YYYYMMDD，用 format="mixed" 兼容两种
    df.index = pd.to_datetime(df.index, format="mixed")
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[DB_START:]
    return df


def _align_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """
    将两个 DataFrame 的 columns（股票代码）取并集，
    缺失部分填 NaN，保持列顺序一致。
    """
    all_cols = df1.columns.union(df2.columns)
    df1 = df1.reindex(columns=all_cols)
    df2 = df2.reindex(columns=all_cols)
    return df1, df2


# -----------------------------------------------------------------------------
# 公开接口：加载单个字段
# -----------------------------------------------------------------------------

def load_field(
    field: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载单个数据字段，自动拼接历史存档与 Database 两个来源。

    Parameters
    ----------
    field : str
        字段名，参考 ARCHIVE_FIELD_MAP 或 DATABASE_FIELD_MAP 的键。
        例如："close_adj", "turn", "pb", "amt", "high", "low"
    start : str, optional
        起始日期，格式 "YYYY-MM-DD"，默认使用 HIST_START
    end : str, optional
        截止日期，格式 "YYYY-MM-DD"，默认使用今天

    Returns
    -------
    pd.DataFrame
        index=DatetimeIndex(日频), columns=股票代码
        两个来源纵向拼接，列（股票代码）取并集，缺失填 NaN

    Notes
    -----
    - 部分字段仅存在于存档（如 close_adj, listing_days, net_profit）
    - 部分字段仅存在于 Database（如 high, low, amt, pe_ttm, dividend_ratio）
    - 共有字段（close, turn, pb）：存档覆盖 2014-2020，Database 覆盖 2021+
    """
    frames = []

    # --- 历史存档来源 ---
    # trade_status 字段含"交易"/"停牌"字符串，不做数值转换
    _STRING_FIELDS = {"trade_status"}

    if field in ARCHIVE_FIELD_MAP:
        try:
            df_hist = _read_archive_csv(
                ARCHIVE_FIELD_MAP[field],
                numeric=(field not in _STRING_FIELDS),
            )
            frames.append(("archive", df_hist))
        except FileNotFoundError as e:
            warnings.warn(str(e))

    # --- Database 来源 ---
    # 所有在 DATABASE_FIELD_MAP 中的字段直接读取；
    # net_profit 仅存档有，其余字段均已覆盖。
    db_field = field if field in DATABASE_FIELD_MAP else None
    if field == "net_profit":
        db_field = None

    if db_field and db_field in DATABASE_FIELD_MAP:
        try:
            df_db = _read_database_csv(DATABASE_FIELD_MAP[db_field])
            frames.append(("database", df_db))
        except FileNotFoundError as e:
            warnings.warn(str(e))

    if not frames:
        raise ValueError(f"[loader] 字段 '{field}' 在存档和 Database 中均未找到。")

    # --- 拼接 ---
    if len(frames) == 1:
        result = frames[0][1]
    else:
        df1 = frames[0][1]
        df2 = frames[1][1]
        df1, df2 = _align_columns(df1, df2)
        result = pd.concat([df1, df2], axis=0)
        # 去重（以防时间段有重叠，优先保留存档数据）
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()

    # --- 时间裁剪 ---
    if start:
        result = result.loc[start:]
    if end:
        result = result.loc[:end]

    return result


# -----------------------------------------------------------------------------
# 公开接口：批量加载多个字段
# -----------------------------------------------------------------------------

def load_data(
    fields: list,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """
    批量加载多个字段，返回字典。

    Parameters
    ----------
    fields : list of str
        字段名列表，例如 ["close_adj", "turn", "pb", "amt"]
    start, end : str, optional
        时间范围

    Returns
    -------
    dict
        {field_name: pd.DataFrame}，每个 DataFrame 的格式同 load_field()
    """
    data = {}
    for field in fields:
        try:
            data[field] = load_field(field, start=start, end=end)
            print(f"[loader] ✓ {field:20s} shape={data[field].shape}")
        except Exception as e:
            warnings.warn(f"[loader] ✗ {field}: {e}")
    return data


# -----------------------------------------------------------------------------
# 公开接口：月度采样
# -----------------------------------------------------------------------------

def to_monthly(df: pd.DataFrame, method: str = "last") -> pd.DataFrame:
    """
    将日度 DataFrame 降采样为月度。

    Parameters
    ----------
    df : pd.DataFrame
        日度数据，index=DatetimeIndex
    method : str
        采样方法："last"（月末值）| "mean"（月均值）| "sum"（月加总）

    Returns
    -------
    pd.DataFrame
        月度数据，index 为每月最后交易日
    """
    if method == "last":
        return df.resample("ME").last()
    elif method == "mean":
        return df.resample("ME").mean()
    elif method == "sum":
        return df.resample("ME").sum()
    else:
        raise ValueError(f"[loader] 不支持的采样方法: {method}")
