# =============================================================================
# strategy/combine_factors.py
# 多因子合成模块
#
# 工作流：
#   1. align_factor_directions()   — 方向对齐（负向因子×-1，统一高分=好股）
#   2. lowdin_orthogonalize()      — Lowdin 正交化（可选，消除因子间共线性）
#   3. calc_rolling_icir_weights() — 滚动 ICIR 权重（12月滚动，动态调整）
#   4. combine_factors()           — 加权合成综合得分
#
# 输出综合得分后，直接传入 backtest/engine.group_return() 进行分组回测，
# 复用现有报告与图表模块。
#
# 参考文献：
#   - 图片资料 1.4节：多因子组合选股及回测（基于A股纺织服装行业，方法论通用）
#   - Lowdin (1950): orthogonalization via eigendecomposition of overlap matrix
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from typing import Optional

from backtest.metrics import calc_ic


# -----------------------------------------------------------------------------
# 1. 方向对齐
# -----------------------------------------------------------------------------

def align_factor_directions(
    factors_dict: dict[str, pd.DataFrame],
    direction_map: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """
    统一因子方向：高得分 = 好股票。

    对 IC 为负的因子乘以 -1，使所有因子均为正向：
    翻转后 G5（最高分组）= 预期收益最高，G1 = 最低。

    Parameters
    ----------
    factors_dict : dict
        {因子名: DataFrame(date × stock)}，来自因子缓存
    direction_map : dict
        {因子名: +1 或 -1}
        +1 = 正向因子（原始 IC > 0，无需翻转）
        -1 = 负向因子（原始 IC < 0，需翻转）

    Returns
    -------
    dict
        方向对齐后的因子字典，结构不变
    """
    aligned = {}
    for name, df in factors_dict.items():
        sign = direction_map.get(name, 1)
        aligned[name] = df * sign
        if sign == -1:
            print(f"  [combine] {name}: 方向翻转（×-1）")
        else:
            print(f"  [combine] {name}: 方向保持（+1）")
    return aligned


# -----------------------------------------------------------------------------
# 2. Lowdin 正交化（可选）
# -----------------------------------------------------------------------------

def lowdin_orthogonalize(
    factors_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Lowdin 正交化：消除因子间多重共线性。

    对每个月截面独立执行：
        S   = F.T @ F              （重叠矩阵，n_factors × n_factors）
        S   = U D U.T              （特征值分解）
        B   = U @ D^(-1/2) @ U.T  （S^(-1/2)）
        F̃   = F @ B                （正交化后因子矩阵）

    正交化后各因子截面相关系数趋近 0，合成时不重复计入相关信息。
    若某月有效股票数 < n_factors × 2，跳过该月（原样返回）。

    Parameters
    ----------
    factors_dict : dict
        方向对齐后的因子字典

    Returns
    -------
    dict
        正交化后的因子字典，结构不变
    """
    names = list(factors_dict.keys())
    n_factors = len(names)

    # 对齐所有因子的日期和股票
    common_idx  = factors_dict[names[0]].index
    common_cols = factors_dict[names[0]].columns
    for df in factors_dict.values():
        common_idx  = common_idx.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)

    aligned = {n: factors_dict[n].reindex(index=common_idx, columns=common_cols)
               for n in names}

    result = {n: aligned[n].copy() for n in names}

    for date in common_idx:
        # 构建截面因子矩阵 F: shape (n_stocks_valid, n_factors)
        rows = [aligned[n].loc[date] for n in names]
        F_df = pd.concat(rows, axis=1)
        F_df.columns = names
        F_df = F_df.dropna()

        if len(F_df) < n_factors * 2:
            continue  # 样本太少，跳过

        F = F_df.values  # (n_stocks, n_factors)

        try:
            M = F.T @ F                           # 重叠矩阵
            eigenvalues, U = np.linalg.eigh(M)   # 对称矩阵用 eigh
            # 数值稳定：过小特征值截断
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
            B = U @ D_inv_sqrt @ U.T              # S^(-1/2)
            F_orth = F @ B                        # 正交化后

            for i, name in enumerate(names):
                result[name].loc[date, F_df.index] = F_orth[:, i]
        except np.linalg.LinAlgError:
            pass  # 奇异矩阵时跳过

    return result


# -----------------------------------------------------------------------------
# 3. 滚动 ICIR 权重
# -----------------------------------------------------------------------------

def calc_rolling_icir_weights(
    factors_dict: dict[str, pd.DataFrame],
    monthly_ret: pd.DataFrame,
    window: int = 12,
) -> pd.DataFrame:
    """
    计算各因子的滚动 ICIR 权重。

    步骤：
        1. 对每个因子逐月计算 Spearman IC（vs 下期收益）
        2. 滚动 window 个月：IR_t = mean(IC) / std(IC)
        3. 负 IR 截断为 0（近期无效的因子权重归零）
        4. 截面归一化：weight = IR / sum(IR)，权重之和为 1

    Parameters
    ----------
    factors_dict : dict
        方向对齐后的因子字典（已确保方向一致）
    monthly_ret : pd.DataFrame
        月度收益率矩阵（来自 engine.calc_monthly_returns()）
    window : int
        滚动窗口（月数），默认 12

    Returns
    -------
    pd.DataFrame
        权重矩阵，index=日期，columns=因子名，每行权重之和为 1
    """
    fwd_ret = monthly_ret.shift(-1)

    ic_dict = {}
    for name, factor in factors_dict.items():
        ic = calc_ic(factor, fwd_ret, method="spearman")
        ic_dict[name] = ic
        print(f"  [combine] {name}: IC 均值={ic.mean():.4f}")

    ic_df = pd.DataFrame(ic_dict).sort_index()

    # 滚动 ICIR
    rolling_mean = ic_df.rolling(window, min_periods=window // 2).mean()
    rolling_std  = ic_df.rolling(window, min_periods=window // 2).std()
    rolling_ir   = rolling_mean / rolling_std.replace(0, np.nan)

    # ── 关键：滞后一期，避免前瞻偏差 ──────────────────────────────
    # IC_T 使用了 return_{T+1}（下期收益）来计算。
    # 若用包含 IC_T 的 rolling ICIR 来决定 T 期的因子权重（也是预测 T+1），
    # 则等于在预测时隐式使用了 return_{T+1} 的信息 → 前瞻偏差。
    # shift(1) 使得 T 期权重由 IC_{T-12}...IC_{T-1} 决定，完全无前瞻。
    rolling_ir = rolling_ir.shift(1)

    # 负 IR 归零（因子近期无效则不使用）
    rolling_ir = rolling_ir.clip(lower=0)

    # 归一化
    row_sum = rolling_ir.sum(axis=1)
    # 若某月所有因子 IR 均为 0（极端情况），退化为等权
    zero_rows = row_sum == 0
    rolling_ir[zero_rows] = 1.0 / len(factors_dict)
    row_sum[zero_rows] = 1.0

    weights = rolling_ir.div(row_sum, axis=0)
    return weights


# -----------------------------------------------------------------------------
# 4. 因子合成
# -----------------------------------------------------------------------------

def combine_factors(
    factors_dict: dict[str, pd.DataFrame],
    weights: "str | pd.DataFrame" = "equal",
) -> pd.DataFrame:
    """
    将多个因子加权合成为综合得分。

    Parameters
    ----------
    factors_dict : dict
        {因子名: DataFrame(date × stock)}，已完成方向对齐
    weights : str or pd.DataFrame
        "equal" → 等权平均（简单均值，忽略 NaN）
        DataFrame → ICIR 滚动权重（来自 calc_rolling_icir_weights()）
                    index=日期，columns=因子名

    Returns
    -------
    pd.DataFrame
        综合得分，index=日期，columns=股票代码
        NaN 表示该股票当月因子值不足，无法打分
    """
    names = list(factors_dict.keys())

    # 对齐所有因子的日期和股票
    common_idx  = factors_dict[names[0]].index
    common_cols = factors_dict[names[0]].columns
    for df in factors_dict.values():
        common_idx  = common_idx.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)

    # 构建三维数组：(n_dates, n_stocks, n_factors)
    factor_stack = np.stack(
        [factors_dict[n].reindex(index=common_idx, columns=common_cols).values
         for n in names],
        axis=2,
    )
    nan_mask = np.isnan(factor_stack)  # (n_dates, n_stocks, n_factors)

    if isinstance(weights, str) and weights == "equal":
        # 等权：nan_mask 处权重为 0，其余均分
        w = np.ones((1, 1, len(names))) / len(names)
        w_broadcast = np.broadcast_to(w, factor_stack.shape).copy().astype(float)
        w_broadcast[nan_mask] = 0.0
        factor_clean = np.where(nan_mask, 0.0, factor_stack)
        numerator   = np.sum(factor_clean * w_broadcast, axis=2)
        denominator = np.sum(w_broadcast, axis=2)

    else:
        # ICIR 权重：(n_dates, n_factors)
        weight_df   = weights.reindex(index=common_idx)[names].fillna(0.0)
        w_arr       = weight_df.values                              # (n_dates, n_factors)
        w_3d        = w_arr[:, np.newaxis, :]                       # (n_dates, 1, n_factors)
        w_broadcast = np.broadcast_to(w_3d, factor_stack.shape).copy().astype(float)
        w_broadcast[nan_mask] = 0.0
        factor_clean = np.where(nan_mask, 0.0, factor_stack)
        numerator   = np.sum(factor_clean * w_broadcast, axis=2)
        denominator = np.sum(w_broadcast, axis=2)

    # 归一化（denominator=0 → NaN）
    composite = np.where(denominator > 0, numerator / denominator, np.nan)

    return pd.DataFrame(composite, index=common_idx, columns=common_cols)
