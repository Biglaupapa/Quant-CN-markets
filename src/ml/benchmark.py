# =============================================================================
# ml/benchmark.py
# 同口径基准：在 ML 的样本外窗口、同分组数下重算既有因子
#
# ── 为什么需要这个模块 ──────────────────────────────────────────────────────
# ML 与既有单因子/合成因子的数字**默认不可比**，差异来自四处，且每一处都能
# 单独造成几个百分点的落差：
#
#   1. 回测区间   既有因子跑 2007-2026 全期；ML 因为要留 84 个月做训练+验证，
#                 样本外只有 2014-2026。牛熊分布完全不同。
#   2. 分组数     既有默认 5 组；GKX/JKP 惯例是 10 组。十分档多空端更极端，
#                 LS 收益天然更高——**这是机械效应，不是能力差异**。
#   3. 组内加权   两边都是等权，这一条本来就一致。
#   4. 股票池     两边都走 build_investable_mask，一致。
#
# 本模块的做法是**重算基准**而不是改既有回测：既有的五分组结果一个数字都不动
# （CLAUDE.md / FACTORS.md 里 62 个因子的记录与验收基准全部保持有效），
# 需要对比时在这里按 ML 的口径重新算一遍。
#
# ── 还有一处不可比，本模块也处理 ────────────────────────────────────────────
# **换手率**。ML 的十分档每月按预测值重排，换手远高于因子排序组合。
# 不算交易成本会系统性高估 ML。`turnover_and_cost()` 对两边一视同仁地算。
# =============================================================================

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = None  # 延迟初始化，见 _cache_dir()


def _cache_dir():
    global CACHE_DIR
    if CACHE_DIR is None:
        from src.config.settings import FACTOR_OUTPUT_DIR
        CACHE_DIR = FACTOR_OUTPUT_DIR / "cache"
    return CACHE_DIR


def load_factor(name: str, prefix: str = "") -> pd.DataFrame:
    """从缓存读一个因子宽表。"""
    p = _cache_dir() / f"{prefix}{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"[benchmark] 缓存不存在：{p}（先跑 python -m src.main）")
    return pd.read_csv(p, index_col=0, parse_dates=True)


def build_composite(pool: Optional[Sequence[str]] = None,
                    directions: Optional[dict] = None) -> pd.DataFrame:
    """重建既有的多因子合成得分（等权）。

    默认用 `main.py` 里的 `COMBINE_FACTORS` 与 `FACTOR_DIRECTIONS`，
    即 v3 版本那 10 个因子（7 微观 + 3 基本面）。
    """
    from src.main import COMBINE_FACTORS, FACTOR_DIRECTIONS
    from src.strategy.combine_factors import align_factor_directions, combine_factors

    if pool is None:
        pool = COMBINE_FACTORS["microstructure"] + COMBINE_FACTORS["fundamental"]
    if directions is None:
        directions = FACTOR_DIRECTIONS

    fac = {n: load_factor(n) for n in pool}
    return combine_factors(align_factor_directions(fac, directions), weights="equal")


# -----------------------------------------------------------------------------
# 换手率与交易成本
# -----------------------------------------------------------------------------

def portfolio_turnover(score: pd.DataFrame, n_groups: int = 10,
                       leg: str = "both") -> pd.Series:
    """逐月计算多空组合的**单边换手率**。

    换手率 = |本月持仓权重 − 上月持仓权重| 之和 / 2，等权口径下即
    「本月新进的股票占比」。多空两腿各算一次后相加（leg="both"）。

    这是 ML 与因子排序**最实质的不可比之处**：预测值每月重估，
    排名跳动远大于缓变的因子值。
    """
    from src.strategy.optimizer import group_by_score

    g = group_by_score(score, n_groups=n_groups)
    legs = {"long": [n_groups], "short": [1], "both": [1, n_groups]}[leg]

    prev = {k: None for k in legs}
    out = {}
    for date in g.index:
        row = g.loc[date]
        tot = 0.0
        for k in legs:
            cur = set(row[row == k].index)
            if prev[k] is not None and (cur or prev[k]):
                union = len(cur | prev[k])
                tot += len(cur - prev[k]) / max(len(cur), 1) if union else 0.0
            prev[k] = cur
        out[date] = tot / len(legs)
    return pd.Series(out).iloc[1:]      # 首月无上期，无法定义


def apply_cost(ls_ret: pd.Series, turnover: pd.Series,
               cost_bps: float) -> pd.Series:
    """把交易成本从多空收益里扣掉。

    `cost_bps` 是**单边**成本（基点）。A 股实务上双边约 30bp
    （佣金 + 印花税 + 冲击成本），故单边 15bp 是中性假设。
    每月成本 = 换手率 × 单边成本 × 2（多空两腿都要交易）。
    """
    idx = ls_ret.index.intersection(turnover.index)
    c = turnover.reindex(idx).fillna(0.0) * (cost_bps / 1e4) * 2.0
    return (ls_ret.reindex(idx) - c).rename(ls_ret.name)


# -----------------------------------------------------------------------------
# 主接口
# -----------------------------------------------------------------------------

def evaluate_score(score: pd.DataFrame,
                   name: str,
                   months: pd.DatetimeIndex,
                   n_groups: int = 10,
                   market: str = "A",
                   cost_bps: Sequence[float] = (0.0, 15.0, 30.0)) -> dict:
    """在指定月份上评估一张打分宽表，口径与 ML 完全一致。

    `months` 传 ML 的样本外月份，保证区间对齐。
    """
    from src.backtest.engine import calc_monthly_returns, group_return
    from src.backtest.metrics import calc_ic, group_summary

    ret_m = calc_monthly_returns(market=market)
    fwd = ret_m.shift(-1)

    s = score.reindex(index=months).dropna(how="all")
    cols = s.columns.intersection(fwd.columns)
    s = s.reindex(columns=cols)
    f = fwd.reindex(index=s.index, columns=cols)

    ic = calc_ic(s, f, method="spearman")
    grp = group_return(s, f, n_groups=n_groups)
    summ = group_summary(grp, freq=12)
    ls = summ.loc["LS"] if "LS" in summ.index else None

    row = {
        "ic": ic.mean(),
        "icir": ic.mean() / ic.std() if ic.std() else np.nan,
        "ls_ann": ls["年化收益"] * 100 if ls is not None else np.nan,
        "ls_sharpe": ls["夏普比率"] if ls is not None else np.nan,
        "ls_mdd": ls["最大回撤"] * 100 if ls is not None else np.nan,
        "win_rate": ls["月度胜率"] * 100 if ls is not None else np.nan,
    }

    # 换手率与成本敏感性
    try:
        to = portfolio_turnover(s, n_groups=n_groups)
        row["turnover"] = to.mean() * 100
        for bp in cost_bps:
            net = apply_cost(grp["LS"], to, bp)
            row[f"ls_sharpe@{bp:.0f}bp"] = (net.mean() / net.std() * np.sqrt(12)
                                            if net.std() else np.nan)
    except Exception as e:
        log.warning("[benchmark] %s 换手率计算失败：%s", name, e)

    return row


def compare(ml_res: dict,
            panel: pd.DataFrame,
            n_groups: int = 10,
            market: str = "A",
            extra_factors: Sequence[str] = ("turnover_20_neutral", "amihud_vwap"),
            ) -> pd.DataFrame:
    """ML 结果 vs 既有基准，一张同口径对比表。

    基准包含：
      · v3 合成（等权 10 因子）—— 主要对手
      · 若干强单因子 —— 参照系
    全部在 ML 的样本外月份、同分组数下重算。
    """
    from src.ml import pipeline

    months = pd.DatetimeIndex(
        ml_res["y_true"].index.get_level_values("date").unique()).sort_values()

    log.info("[benchmark] 同口径区间 %s ~ %s（%d 个月），%d 分组",
             f"{months.min():%Y-%m}", f"{months.max():%Y-%m}", len(months), n_groups)

    rows = {}
    # 既有基准
    try:
        rows["【基准】v3合成等权"] = evaluate_score(
            build_composite(), "composite", months, n_groups, market)
    except Exception as e:
        log.warning("[benchmark] 合成因子重算失败：%s", e)
    for fn in extra_factors:
        try:
            rows[f"【基准】{fn}"] = evaluate_score(
                load_factor(fn), fn, months, n_groups, market)
        except Exception as e:
            log.warning("[benchmark] %s 重算失败：%s", fn, e)

    # ML 各模型
    from src.ml.dataset import to_wide
    for mn, yhat in ml_res["preds"].items():
        try:
            rows[mn] = evaluate_score(to_wide(yhat), mn, months, n_groups, market)
        except Exception as e:
            log.warning("[benchmark] %s 评估失败：%s", mn, e)

    df = pd.DataFrame(rows).T
    return df.sort_values("ls_sharpe", ascending=False)
