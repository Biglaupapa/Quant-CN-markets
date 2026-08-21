# =============================================================================
# ml/dataset.py
# 宽表因子 → ML 长面板
#
# 把 output/cache/ 下 N 个「月末 × 股票」的因子宽表，拼成一张
# MultiIndex(date, code) 的长面板，附上标签与回测所需的辅助列。
#
# ── 与 GKX 的三处一致 ────────────────────────────────────────────────────────
#   1. 特征做**截面 rank 标准化到 [-1,1]**（Kelly-Pruitt-Su 2019 /
#      Freyberger-Neuhierl-Weber 2020 的做法，GKX 沿用）
#   2. 缺失值用**每月截面中位数**填充（rank 之后即 0）
#   3. 标签是**下月超额收益的原始数值**，不做任何秩变换
#
# ── 第 3 点为什么重要 ────────────────────────────────────────────────────────
# R²_oos 的定义是 1 − Σ(y−ŷ)²/Σy²，分母**不去均值**（以 0 为基准）。
# 一旦把标签也做了秩变换，这个比值就失去「相对于预测 0」的经济含义，
# 算出来的数字无法与 GKX 的 0.3%~0.4% 对照。所以标签必须保持原始收益率。
#
# ── 为什么 rank 标准化对本项目尤其合适 ──────────────────────────────────────
# 既有 `factors/base.py: preprocess` 是「winsorize(±3σ) → standardize」，
# 在重尾因子上会留下极端值：amihud 的截面最大值中位数 5.17、最高 66.6，
# est_peg 最高 48.9（裁剪后 σ 大幅缩小，再标准化就把边界值放大了）。
# 这是既有口径，**不改**——而 rank→[-1,1] 天然免疫，正好绕开该问题。
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config.settings import FACTOR_OUTPUT_DIR

log = logging.getLogger(__name__)

CACHE_DIR = FACTOR_OUTPUT_DIR / "cache"

# 非特征列（回测与加权用，不进模型）
AUX_COLS = ["y", "ret_fwd", "me"]


# -----------------------------------------------------------------------------
# 因子发现
# -----------------------------------------------------------------------------

def list_cached_factors(market: str = "A") -> list[str]:
    """列出缓存中可用的因子名（A 股不含 hk_ 前缀）。"""
    prefix = "hk_" if market == "HK" else ""
    names = []
    for p in sorted(CACHE_DIR.glob("*.csv")):
        n = p.stem
        if market == "HK":
            if n.startswith("hk_"):
                names.append(n[3:])
        elif not n.startswith("hk_"):
            names.append(n)
    return names


# -----------------------------------------------------------------------------
# 截面变换
# -----------------------------------------------------------------------------

def rank_to_unit(df: pd.DataFrame) -> pd.DataFrame:
    """逐月截面秩标准化到 [-1, 1]。

        rank_pct ∈ (0,1]  →  2·rank_pct − 1 ∈ (-1,1]

    NaN 保持 NaN（由调用方决定怎么填），不参与排名。
    单调变换，因此不改变任何基于排序的结论（IC、分组）。
    """
    r = df.rank(axis=1, pct=True)
    return 2.0 * r - 1.0


def fill_cross_median(df: pd.DataFrame) -> pd.DataFrame:
    """用每月截面中位数填充缺失（GKX 做法）。

    rank 之后截面中位数恒为 0（对称分布在 [-1,1]），所以这一步等价于填 0；
    这里仍按中位数写，是为了在 `rank_transform=False` 时也语义正确。
    """
    med = df.median(axis=1)
    return df.apply(lambda col: col.fillna(med), axis=0)


# -----------------------------------------------------------------------------
# 标签
# -----------------------------------------------------------------------------

def _load_rf_monthly() -> Optional[pd.Series]:
    """月度无风险利率（ff3_monthly.csv 的 Rf 列）。"""
    p = Path("/Users/louis/MyProjects/Database/data/factors/ff3_monthly.csv")
    if not p.exists():
        log.warning(f"[dataset] FF3 月度文件不存在：{p}，标签将用原始收益而非超额收益")
        return None
    ff3 = pd.read_csv(p, index_col=0, parse_dates=True)
    if "Rf" not in ff3.columns:
        return None
    return ff3["Rf"]


def build_label(start: Optional[str] = None,
                end: Optional[str] = None,
                market: str = "A",
                excess: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造标签：下月收益（可选超额）。

    Returns
    -------
    (y, ret_fwd)
        y       : 下月**超额**收益（excess=True）或原始收益，用于训练与 R²_oos
        ret_fwd : 下月**原始**收益，用于组合回测（与既有单因子流程一致）

    时序对齐与既有框架完全一致：T 月末的特征预测 T+1 月收益，
    即 `monthly_ret.shift(-1)`。见 main.py 的 `fwd_ret = monthly_ret.shift(-1)`。
    """
    from src.backtest.engine import calc_monthly_returns

    ret_m = calc_monthly_returns(start=start, end=end, market=market)
    ret_fwd = ret_m.shift(-1)

    if not excess:
        return ret_fwd.copy(), ret_fwd

    rf = _load_rf_monthly()
    if rf is None:
        return ret_fwd.copy(), ret_fwd

    # Rf 对齐到 ret_fwd 的月份：T 行放的是 T+1 月收益，故减 T+1 月的 Rf
    rf_fwd = rf.reindex(ret_m.index).shift(-1)
    y = ret_fwd.sub(rf_fwd, axis=0)
    return y, ret_fwd


# -----------------------------------------------------------------------------
# 主接口
# -----------------------------------------------------------------------------

def build_panel(
    factors: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "A",
    rank_transform: bool = True,
    fill_missing: bool = True,
    max_missing_ratio: float = 0.75,
    min_features: int = 5,
    dtype=np.float32,
) -> pd.DataFrame:
    """构建 ML 长面板。

    Parameters
    ----------
    factors : 因子名列表；None = 缓存里的全部
    rank_transform : 截面 rank → [-1,1]（GKX 做法，默认开）
    fill_missing : 用截面中位数填充缺失（GKX 做法，默认开）
    max_missing_ratio : 缺失率超过此值的**因子整列剔除**（GKX 用 0.25，
        即保留缺失 <25% 的；A 股因为 2005~2010 上市公司少、且部分因子
        需要 36 个月滚动窗口，缺失率天然偏高，这里放宽到 0.75）
    min_features : 一行至少要有几个非空特征才保留（在填充**之前**判定）

    Returns
    -------
    pd.DataFrame
        MultiIndex(date, code)，列 = 各因子 + ["y", "ret_fwd", "me"]
    """
    names = list(factors) if factors is not None else list_cached_factors(market)
    prefix = "hk_" if market == "HK" else ""

    # ── 1. 读取因子宽表 ──────────────────────────────────────────
    wide: dict[str, pd.DataFrame] = {}
    for n in names:
        p = CACHE_DIR / f"{prefix}{n}.csv"
        if not p.exists():
            log.warning(f"[dataset] 缓存缺失，跳过：{p.name}")
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]
        wide[n] = df
    if not wide:
        raise ValueError("[dataset] 没有可用的因子缓存，请先跑 python -m src.main")

    # ── 2. 标签 ─────────────────────────────────────────────────
    y, ret_fwd = build_label(start=start, end=end, market=market)

    # ── 3. 统一 index / columns ─────────────────────────────────
    #
    # ⚠️ 这里**不能取因子日期的交集**。若干因子需要长滚动窗口才有值
    # （ap_beta1~5 与 ps_liq_beta 是 36 月、ret_36_13 是 36 月），
    # 取交集会把整个面板的起点拖到 2010-02，白白丢掉 2007~2010 三年
    # （约 15 万个可训练格子）。
    #
    # 正确做法与 GKX 一致：**以标签的月份为准**，某个因子在早期没有值
    # 就当缺失，交给第 7 步的截面中位数填充。填充后该因子在那些月份是
    # 常数列——对树模型没有可切分点、对线性模型被截距吸收，无害。
    idx = y.index.sort_values()
    cols = y.columns
    for df in wide.values():
        cols = cols.union(df.columns)
    cols = cols.intersection(y.columns).sort_values()

    y = y.reindex(index=idx, columns=cols)
    ret_fwd = ret_fwd.reindex(index=idx, columns=cols)

    # ── 4. 缺失率过滤 + 截面 rank ────────────────────────────────
    kept, dropped = [], []
    for n, df in wide.items():
        df = df.reindex(index=idx, columns=cols)
        miss = float(df.isna().to_numpy().mean())
        if miss > max_missing_ratio:
            dropped.append((n, miss))
            continue
        wide[n] = rank_to_unit(df) if rank_transform else df
        kept.append(n)

    if dropped:
        log.info("[dataset] 因缺失率过高剔除 %d 个因子：%s", len(dropped),
                 ", ".join(f"{n}({m:.0%})" for n, m in sorted(dropped, key=lambda x: -x[1])))
    if not kept:
        raise ValueError("[dataset] 所有因子都被缺失率过滤掉了，请调高 max_missing_ratio")

    # ── 5. 辅助列：流通市值（市值加权组合用）──────────────────────
    me = None
    if market == "A":
        try:
            from src.data.loader import load_data, to_monthly
            mv = load_data(["neg_market_value"], start=start, end=end)["neg_market_value"]
            me = to_monthly(mv, method="last").reindex(index=idx, columns=cols)
        except Exception as e:
            log.warning(f"[dataset] 流通市值加载失败（市值加权将不可用）：{e}")

    # ── 6. 堆叠成长面板 ─────────────────────────────────────────
    # 先按「标签非空」筛掉不可训练的格子，能把内存砍掉一半以上
    valid = y.notna()
    n_feat_valid = sum(wide[n].notna() for n in kept)
    valid &= (n_feat_valid >= min_features)

    log.info("[dataset] 可用格子 %s / %s（%.1f%%）",
             f"{int(valid.to_numpy().sum()):,}", f"{valid.size:,}",
             100 * valid.to_numpy().mean())

    pieces = {}
    for n in kept:
        pieces[n] = wide[n].where(valid).stack(future_stack=True)
    pieces["y"] = y.where(valid).stack(future_stack=True)
    pieces["ret_fwd"] = ret_fwd.where(valid).stack(future_stack=True)
    if me is not None:
        pieces["me"] = me.where(valid).stack(future_stack=True)

    panel = pd.DataFrame(pieces)
    panel = panel[panel["y"].notna()]
    panel.index.names = ["date", "code"]

    # ── 7. 缺失填充（只填特征，不填标签）─────────────────────────
    if fill_missing:
        med = panel.groupby(level="date")[kept].transform("median")
        panel[kept] = panel[kept].fillna(med)
        # 整月全空的因子中位数仍是 NaN → 填 0（rank 口径下即截面中位）
        panel[kept] = panel[kept].fillna(0.0)

    for c in panel.columns:
        if c != "me":
            panel[c] = panel[c].astype(dtype)

    log.info("[dataset] 面板 %s 行 × %d 特征，%s ~ %s，%d 只股票",
             f"{len(panel):,}", len(kept),
             panel.index.get_level_values(0).min().date(),
             panel.index.get_level_values(0).max().date(),
             panel.index.get_level_values(1).nunique())

    panel.attrs["features"] = kept
    panel.attrs["dropped"] = dropped
    return panel


def feature_cols(panel: pd.DataFrame) -> list[str]:
    """从面板取特征列名（排除辅助列）。"""
    if "features" in panel.attrs:
        return list(panel.attrs["features"])
    return [c for c in panel.columns if c not in AUX_COLS]


def to_wide(series: pd.Series) -> pd.DataFrame:
    """长面板的一列 → 「月末 × 股票」宽表。

    这是把 ML 预测接回既有回测链路的唯一接口：
        pred_wide = to_wide(panel_pred["yhat"])
        group_return(pred_wide, fwd_ret, n_groups=10)
    """
    w = series.unstack(level="code")
    w.index.name = None
    return w
