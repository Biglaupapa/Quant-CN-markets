# =============================================================================
# ml/evaluate.py
# ML 专属评估：R²_oos / Diebold-Mariano / 变量重要性 / 张成检验
#
# 既有 backtest/metrics.py 已有 IC、ICIR、Sharpe、最大回撤，那些照旧复用。
# 本模块只补 ML 资产定价文献特有、而框架里没有的四件东西。
#
# 三个指标各管一件事，缺一不可：
#   R²_oos    预测得**准不准**（数值精度）   ← 本模块
#   IC/ICIR   排序得**稳不稳**（秩相关）     ← metrics.calc_ic
#   LS/Sharpe 这个预测**值多少钱**（经济价值）← metrics.group_summary
# 一个模型完全可能 IC 漂亮但 R²_oos 为负——只要它把收益的量级系统性预测偏大。
# =============================================================================

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# =============================================================================
# 一、样本外预测 R²
# =============================================================================

def r2_oos(y_true, y_pred) -> float:
    """Gu-Kelly-Xiu (2020) 式 (17) 的样本外预测 R²。

        R²_oos = 1 − Σ(y − ŷ)² / Σ y²

    ── 分母不去均值，这是与普通 R² 最本质的区别 ──────────────────────────────
    普通 R² 用 Σ(y − ȳ)²，即拿**历史均值**当基准。GKX 刻意改成 Σy²，
    即拿 **0** 当基准，原文的理由是：个股月度收益的历史均值是一个极度嘈杂的
    估计量，用它当基准会人为抬高 R²。而「预测超额收益为 0」是一个诚实、
    无参数、且有经济含义的基准。因此 R²_oos > 0 的含义是：

        「我的模型比『闭着眼睛说所有股票的超额收益都是 0』要好。」

    ── 数值有多小才算正常 ──────────────────────────────────────────────────
    GKX 在美股全样本上的月度 R²_oos 大致是（近似值）：
        OLS 全特征 ≈ −3.5%（负的！高维无正则的线性模型在此问题上灾难性过拟合）
        OLS-3 ≈ +0.16% | ENet ≈ +0.11% | PLS/PCR ≈ +0.27%
        RF ≈ +0.33% | GBRT ≈ +0.34% | NN3 ≈ +0.40%（最优）
    **0.4% 就是这个领域的天花板级结果**，因为个股月度收益 99% 以上是噪音。
    不要拿它跟别处见过的 R²=0.3 之类的数字比——那些是完全不同的问题结构。

    ── 与 IC 的粗略换算 ────────────────────────────────────────────────────
    对校准良好的预测，R²_oos ≈ IC²。本框架合成因子 IC≈0.11 → 上限约 1.2%。
    注意是**上限**：R²_oos 对尺度和偏差敏感（预测值整体放大 10 倍就会变成负数），
    IC 则完全不敏感（秩相关）。二者是互补而非替代关系。
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return np.nan
    y, p = y[m], p[m]
    denom = float((y ** 2).sum())
    if denom == 0:
        return np.nan
    return 1.0 - float(((y - p) ** 2).sum()) / denom


def r2_oos_by(df: pd.DataFrame, y_col: str, pred_col: str,
              by: str) -> pd.Series:
    """按某个维度分组算 R²_oos（如逐年、按市值分档）。"""
    return df.groupby(by).apply(
        lambda g: r2_oos(g[y_col], g[pred_col]), include_groups=False)


def size_subsample_r2(df: pd.DataFrame, y_col: str, pred_col: str,
                      me_col: str = "me", date_level: str = "date",
                      p: float = 0.3) -> dict:
    """全样本 / 大盘 / 小盘三档 R²_oos（GKX 与德国项目都报这三档）。

    每月按流通市值取前 p 与后 p，考察模型的预测力是否集中在某个规模域。
    A 股小市值溢价极强，这一拆分尤其有信息量。
    """
    out = {"全样本": r2_oos(df[y_col], df[pred_col])}
    if me_col not in df.columns:
        return out
    d = df[df[me_col].notna()]
    rk = d.groupby(level=date_level)[me_col].rank(pct=True)
    out["大盘(前30%)"] = r2_oos(d.loc[rk >= 1 - p, y_col], d.loc[rk >= 1 - p, pred_col])
    out["小盘(后30%)"] = r2_oos(d.loc[rk <= p, y_col], d.loc[rk <= p, pred_col])
    return out


# =============================================================================
# 二、Diebold-Mariano 检验
# =============================================================================

def dm_test(e1, e2, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano 预测精度检验，Newey-West 修正。

        d_t = |e1_t|^power − |e2_t|^power
        DM  = d̄ / se(d̄)

    统计量**为正**表示模型 2（e2）比模型 1 更准（模型1 的损失更大）。

    ── 为什么这个检验是必需的 ──────────────────────────────────────────────
    各模型的 R²_oos 都挤在 0.1%~0.4% 这个窄带里，「RF 的 0.33% 比 PLS 的
    0.27% 更好」这句话本身没有统计意义——差距可能完全来自抽样噪音。
    不做 DM 检验，模型排序的结论站不住。

    Returns
    -------
    (dm_stat, p_value)
    """
    from scipy import stats

    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    m = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[m], e2[m]
    if len(e1) < 3:
        return np.nan, np.nan

    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    dc = d - dbar

    # Newey-West 长期方差（h−1 阶自协方差，Bartlett 权重）
    gamma0 = float((dc ** 2).sum()) / n
    var = gamma0
    for lag in range(1, h):
        g = float((dc[lag:] * dc[:-lag]).sum()) / n
        var += 2.0 * (1.0 - lag / h) * g
    var /= n
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan

    dm = dbar / np.sqrt(var)
    pv = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return float(dm), float(pv)


def dm_matrix(preds: dict[str, np.ndarray], y_true,
              h: int = 1, power: int = 2) -> pd.DataFrame:
    """两两 DM 统计量矩阵，带显著性星号。

    读法：**正值表示「列模型」优于「行模型」**。
    """
    y = np.asarray(y_true, dtype=np.float64)
    names = list(preds)
    out = pd.DataFrame("", index=names, columns=names, dtype=object)
    for a in names:
        for b in names:
            if a == b:
                out.loc[a, b] = "—"
                continue
            dm, pv = dm_test(y - preds[a], y - preds[b], h=h, power=power)
            if not np.isfinite(dm):
                out.loc[a, b] = "n/a"
                continue
            star = "***" if pv <= 0.01 else "**" if pv <= 0.05 else "*" if pv <= 0.10 else ""
            out.loc[a, b] = f"{dm:.2f}{star}"
    return out


# =============================================================================
# 三、变量重要性（置零法）
# =============================================================================

def variable_importance(model, X: pd.DataFrame, y,
                        features: Sequence[str],
                        baseline: Optional[float] = None) -> pd.Series:
    """GKX 的置零法变量重要性。

    把某个预测变量的**全部取值置零**，测 R²_oos 的下降幅度，
    再把各变量的绝对下降幅度归一化到和为 1。

    与 sklearn 的 `feature_importances_` 的区别：
      · 置零法是**模型无关**的，线性模型和树模型可以放在同一张图上比
      · 它衡量的是「对预测力的贡献」，而非「树里被用了多少次」

    注：特征已做 rank→[-1,1] 变换，截面中位数恰为 0，
    因此「置零」在语义上等价于「把该变量替换成截面中位数」，
    即抹掉它的截面区分度而不引入人为的极端值。
    """
    from src.ml.models import predict

    Xa = X[list(features)].to_numpy(dtype=np.float64)
    if baseline is None:
        baseline = r2_oos(y, predict(model, Xa))

    drops = {}
    for j, f in enumerate(features):
        Xz = Xa.copy()
        Xz[:, j] = 0.0
        drops[f] = baseline - r2_oos(y, predict(model, Xz))

    s = pd.Series(drops)
    tot = s.abs().sum()
    return (s.abs() / tot).sort_values(ascending=False) if tot > 0 else s


def _monthly_ic(pred: np.ndarray, y_rank: pd.Series, dates: pd.Index) -> float:
    """月均截面 Spearman IC；y_rank 为预先按月算好的标签秩。"""
    pr = pd.Series(pred, index=y_rank.index).groupby(dates).rank()
    return pr.groupby(dates).corr(y_rank).mean()


def importance_drops(model, X: np.ndarray, y: np.ndarray, dates: pd.Index,
                     features: Sequence[str]) -> pd.DataFrame:
    """
    置零法变量重要性的**原始下降量**（未归一化），两种口径：

    - `r2`：置零后面板 R² 的下降。**GKX（2020）§2.9 / §3.3 的口径**，且 GKX 在
      「each training sample」上计算、再对所有训练样本取平均——调用方应传入训练集
    - `ic`：置零后月均截面 Spearman IC 的下降。**补充口径，无直接文献依据**：
      本框架按预测排序做多空，排序能力是策略实际用到的信息；R² 以 0 为基准、对预测整体
      水平敏感，置零某变量若使预测整体平移，R² 会大降而排序几乎不变（2026-10-03 roll_spread 案例）

    特征已做截面 rank→[-1,1]，置零 = 换成截面中位数。
    """
    from src.ml.models import predict

    dates = pd.Index(dates)
    y_rank = pd.Series(y, index=pd.RangeIndex(len(y))).groupby(dates).rank()
    y_rank.index = pd.RangeIndex(len(y))
    dates = pd.Index(np.asarray(dates))
    p0 = predict(model, X)
    r0, ic0 = r2_oos(y, p0), _monthly_ic(p0, y_rank, dates)
    rows = {}
    for j, f in enumerate(features):
        Xz = X.copy()
        Xz[:, j] = 0.0
        pz = predict(model, Xz)
        rows[f] = {"r2": r0 - r2_oos(y, pz), "ic": ic0 - _monthly_ic(pz, y_rank, dates)}
    return pd.DataFrame(rows).T


def normalize_importance(drops: pd.Series) -> pd.Series:
    """负下降（置零后反而更好）记 0，再归一化到和为 1（GKX：normalized to sum to one）。"""
    d = drops.clip(lower=0)
    tot = d.sum()
    return (d / tot).sort_values(ascending=False) if tot > 0 else d


# =============================================================================
# 四、张成检验（spanning regression）
# =============================================================================

def spanning_test(target_ls: pd.Series,
                  factor_ls: pd.DataFrame,
                  nw_lags: int = 6) -> pd.DataFrame:
    """把 ML 多空收益对既有因子多空收益做时序回归，看 α 是否显著为正。

        r^ML_t = α + Σ_k β_k · r^factor_k_t + ε_t

    这是回答「ML 到底带来了新东西吗」最严格的方式，比直接比较年化收益
    有说服力得多，也是审稿人一定会问的：

      · α 显著为正        → ML 捕捉到了既有因子**张不成**的信息，非线性有真实增量
      · α ≈ 0 且 β 集中在 size / amihud 上
                          → ML 只是用复杂方式重新拟合了小市值与流动性溢价，
                            没有新信息，那么 10 因子等权就够了

    标准误用 Newey-West（月频收益有自相关与异方差）。
    """
    import statsmodels.api as sm

    idx = target_ls.dropna().index.intersection(factor_ls.dropna(how="all").index)
    if len(idx) < 24:
        raise ValueError(f"[evaluate] 张成检验样本不足（{len(idx)} 个月）")

    yv = target_ls.loc[idx].astype(float)
    Xv = sm.add_constant(factor_ls.loc[idx].astype(float), has_constant="add")
    res = sm.OLS(yv, Xv, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    out = pd.DataFrame({
        "系数": res.params,
        "t值(NW)": res.tvalues,
        "p值": res.pvalues,
    })
    out.attrs["r2"] = res.rsquared
    out.attrs["n"] = int(res.nobs)
    # α 是月度值，年化便于与年化收益对读
    out.attrs["alpha_ann"] = float(res.params.get("const", np.nan)) * 12
    return out


# =============================================================================
# 五、汇总
# =============================================================================

def summarize(results: dict[str, dict]) -> pd.DataFrame:
    """把各模型的评估结果汇成一张表。

    `results` 形如 {模型名: {"r2_oos":…, "ic":…, "icir":…, "ls_ann":…, …}}
    """
    df = pd.DataFrame(results).T
    order = ["r2_oos", "r2_large", "r2_small", "ic", "icir",
             "ls_ann", "ls_sharpe", "ls_mdd", "win_rate"]
    cols = [c for c in order if c in df.columns] + \
           [c for c in df.columns if c not in order]
    return df[cols].sort_values("r2_oos", ascending=False)
