# =============================================================================
# ml/pipeline.py
# 滚动训练 → 样本外预测 → 评估，一条龙
#
# 产出的核心是一张「月末 × 股票」的预测宽表 ŷ，直接喂给既有回测链路：
#
#     pred_wide = to_wide(preds["GBRT"])
#     grp = group_return(pred_wide, fwd_ret, n_groups=10)   # backtest/engine.py
#     ic  = calc_ic(pred_wide, fwd_ret, method="spearman")  # backtest/metrics.py
#
# 因此 ML 结果与既有 62 个单因子走的是**完全相同**的评估口径，天然可比。
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.ml.cv import RollingWindowCV, assert_no_leakage
from src.ml.dataset import feature_cols, to_wide
from src.ml.models import ModelSpec, default_specs, fit_with_validation, predict

log = logging.getLogger(__name__)


def _slice(panel: pd.DataFrame, months: pd.DatetimeIndex,
           feats: list[str]) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """按月份取出 (X, y, index)。"""
    sub = panel[panel.index.get_level_values("date").isin(months)]
    return (sub[feats].to_numpy(dtype=np.float64),
            sub["y"].to_numpy(dtype=np.float64),
            sub.index)


def mask_immature_features(X_fit: np.ndarray, dates_fit: pd.Index, feats: list[str],
                           min_months: int, *arrays: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    """训练样本中有效历史不足 `min_months` 个月的特征，在本窗口整列置 0（= 缺失填充值）。

    「有效月」= 该月超过一半股票的值非 0（面板 rank→[-1,1] 后缺失即 0）。只看拟合用的
    训练样本（train+val），不引入前视。（待办 #23，2026-10-05）
    起因：rd_sale / rd_me / rd_at 自 2018-10 才有值（2018 年起利润表单列研发费用）。
    训练期到 2018-12 的窗口里它们只有 3–4 个月有值，线性模型据此估出过大系数
    （rd_sale 0.48，其余特征 ≤ 0.08），2019 年测试期全面有值后预测失控（单窗 R²_oos −589%）。
    返回 (被置 0 的特征, 置 0 后的各数组)；置 0 而非删列，保持列结构与重要性对齐。
    """
    covered = pd.DataFrame(X_fit != 0, columns=feats).groupby(np.asarray(dates_fit)).mean() > 0.5
    immature = [f for f, k in covered.sum().items() if k < min_months]
    if not immature:
        return [], list(arrays)
    jj = [feats.index(f) for f in immature]
    out = []
    for a in arrays:
        a = a.copy()
        a[:, jj] = 0.0
        out.append(a)
    return immature, out


def _fit_calibration(pred_va: np.ndarray, y_va: np.ndarray) -> tuple[float, float]:
    """在**验证集**上拟合一元映射 ŷ' = a + b·ŷ。

    树模型的预测幅度会系统性偏大：叶子值记录的是训练期的截面离散度，
    直接搬到测试期就过度自信。实测（2026-08-20）：

        模型      Pearson相关   R²_oos     最优缩放 b
        ENet        0.046      +0.31%       0.53
        GBRT        0.072      −5.05%       0.26
        LGBM        0.074      −9.68%       0.19

    LGBM 的信息含量其实**最高**（相关 0.074，是 ENet 的 1.6 倍），
    但预测幅度是应有的 5 倍，于是 R²_oos 被砸到 −9.68%。
    线性模型（尤其带 L1/L2 正则的 ENet）天然向 0 收缩，尺度正好，所以没这问题。

    ── 为什么在验证集上拟合而不是测试集 ────────────────────────────────────
    验证集本来就用于选超参，模型在那一段上是「样本外」的，用它拟合校准
    **不引入前视偏差**。若在测试集上拟合就是 oracle，只能当诊断（见
    `evaluate.py` 的 `r2_oos_oracle` 列），不能当性能指标。

    ── 校准不改变任何排序 ──────────────────────────────────────────────────
    b > 0 时这是单调变换，IC、分组、多空收益**逐位不变**，只有 R²_oos 会变。
    所以它修的是「预测的量级」，不是「预测的方向」。
    b ≤ 0 说明模型在验证集上方向就是反的，此时退回不校准（a=0, b=1），
    避免把一个已经失效的模型再翻转一次。

    ── ⛔️ 为什么默认关闭（2026-08-20 实测结论）──────────────────────────────
    **这个方案不奏效，开启反而更差**：

        模型     r2_oos(关)   r2_oos(开)   r2_oracle
        ENet      +0.311      −1.164        +0.316
        LGBM      −4.858      −7.993        +0.423

    根因是**验证段与测试段对最优尺度的判断方向相反**：
    验证集拟合出的 b 中位数 1.21（认为预测偏小、该放大），
    而测试集的最优 b 是 0.19~0.53（预测偏大、该缩小）。
    照验证集调，正好调反。

    结论：树模型的失准**不是一个固定的尺度偏差，而是随时间变化的**，
    紧邻的 24 个月验证窗口预测不了下一年的最优尺度。

    「设计意图」那部分倒是验证通过了——IC 与夏普在开关前后逐位相同
    （0.1133/0.1133、2.034/2.034），确实不改排序。

    代码保留不删，因为这个负面结果本身有价值：它说明想修 R²_oos 不能靠
    简单重标定，得从模型侧入手（例如对树模型的叶子值做收缩、或改用
    分位数目标）。要重新尝试时把 `ML_CONFIG["calibrate"]` 打开即可。
    """
    p = np.asarray(pred_va, dtype=np.float64)
    y = np.asarray(y_va, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() < 100:
        return 0.0, 1.0
    p, y = p[m], y[m]
    var = float(((p - p.mean()) ** 2).sum())
    if var <= 0:
        return 0.0, 1.0
    b = float(((p - p.mean()) * (y - y.mean())).sum() / var)
    if b <= 0:
        return 0.0, 1.0
    return float(y.mean() - b * p.mean()), b


def _r2_oracle(y: np.ndarray, p: np.ndarray) -> float:
    """在测试集上拟合最优 a、b 后的 R²_oos —— **诊断用，有前视偏差**。

    回答「负 R² 是信息问题还是校准问题」。它是任何单调线性重标定所能达到的
    R² 上界，所以：oracle 仍为负 = 信息真的不够；oracle 为正 = 只是尺度不对。
    """
    from src.ml.evaluate import r2_oos
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 100:
        return float("nan")
    y, p = y[m], p[m]
    var = float(((p - p.mean()) ** 2).sum())
    if var <= 0:
        return float("nan")
    b = float(((p - p.mean()) * (y - y.mean())).sum() / var)
    a = float(y.mean() - b * p.mean())
    return r2_oos(y, a + b * p)


def run(panel: pd.DataFrame,
        specs: Optional[Sequence[ModelSpec]] = None,
        cv: Optional[RollingWindowCV] = None,
        loss: str = "huber",
        fast: bool = True,
        calibrate: bool = True,
        cooldown: float = 0.0,
        verbose: bool = True,
        importance_model: Optional[str] = None,
        min_feature_months: int = 0) -> dict:
    """滚动训练全部模型，返回样本外预测。

    Parameters
    ----------
    calibrate : bool
        是否在验证集上拟合一元映射把预测放回收益量纲（见 `_fit_calibration`）。
        **不改变任何排序**，只影响 R²_oos。默认开启。
    min_feature_months : int
        特征在训练样本中至少要有的有效月数，不足则本窗置 0（见 `mask_immature_features`）。0 = 不检查。
    cooldown : float
        每个滚动窗口训练完后的散热间歇（秒）。默认 0 = 不停。
        纯粹是给机器降温用的，**不影响任何计算结果**——窗口之间本就无状态。
        线程预算（多少核）在 `src/config/compute.py`，与此正交：
        限线程降峰值功耗，间歇降持续积热，两者可叠加。

    Returns
    -------
    dict
        preds   : {模型名: pd.Series}  MultiIndex(date, code) 的样本外 ŷ
        y_true  : pd.Series            对应的真实标签
        params  : {模型名: DataFrame}  各窗口选中的超参
        cal     : {模型名: DataFrame}  各窗口的校准系数 a、b（calibrate=True 时）
        timing  : {模型名: 秒}
        importance : {窗口号: DataFrame[r2, ic]}  importance_model 在**该窗口训练集**上的
                     置零法原始下降量（GKX 2020：within each training sample，再跨样本平均）
    """
    feats = feature_cols(panel)
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    cv = cv or RollingWindowCV()
    specs = list(specs) if specs is not None else default_specs(feats, fast=fast)

    splits = list(cv.split(dates))
    if verbose:
        log.info("[pipeline] %d 个特征，%d 个月，%d 个滚动窗口，%d 个模型",
                 len(feats), len(dates), len(splits), len(specs))
        log.info("[pipeline] 样本外区间：%s ~ %s（%d 个月）",
                 f"{splits[0].test[0]:%Y-%m}", f"{splits[-1].test[-1]:%Y-%m}",
                 sum(len(s.test) for s in splits))

    preds: dict[str, list] = {s.name: [] for s in specs}
    params: dict[str, list] = {s.name: [] for s in specs}
    timing: dict[str, float] = {s.name: 0.0 for s in specs}
    cal_ab: dict[str, list] = {s.name: [] for s in specs}
    y_parts: list[pd.Series] = []
    importance: dict[int, pd.DataFrame] = {}

    for wi, sp in enumerate(splits, 1):
        assert_no_leakage(sp)                       # 每个窗口都验一次，成本可忽略

        X_tr, y_tr, ix_tr = _slice(panel, sp.train, feats)
        X_va, y_va, ix_va = _slice(panel, sp.val, feats)
        X_te, y_te, ix_te = _slice(panel, sp.test, feats)
        if len(X_te) == 0:
            continue
        if min_feature_months:
            immature, (X_tr, X_va, X_te) = mask_immature_features(
                np.vstack([X_tr, X_va]),
                ix_tr.get_level_values("date").append(ix_va.get_level_values("date")),
                feats, min_feature_months, X_tr, X_va, X_te)
            if immature and verbose:
                log.info("    历史不足 %d 月、本窗置 0：%s", min_feature_months, immature)

        y_parts.append(pd.Series(y_te, index=ix_te))
        if verbose:
            log.info("─ 窗口 %d/%d  %s  训练%s 验证%s 测试%s",
                     wi, len(splits), f"测试 {sp.test[0]:%Y-%m}~{sp.test[-1]:%Y-%m}",
                     f"{len(X_tr):,}", f"{len(X_va):,}", f"{len(X_te):,}")

        for spec in specs:
            cols = spec.features or feats
            jj = [feats.index(c) for c in cols]
            t0 = time.time()
            try:
                model, best_p, vloss = fit_with_validation(
                    spec, X_tr[:, jj], y_tr, X_va[:, jj], y_va, loss=loss)
                yhat = predict(model, X_te[:, jj])
                if calibrate:
                    a, b = _fit_calibration(predict(model, X_va[:, jj]), y_va)
                    yhat = a + b * yhat
                    cal_ab[spec.name].append({"窗口": wi, "a": a, "b": b})
            except Exception as e:
                log.warning("  ✗ %s 窗口%d 失败：%s", spec.name, wi, e)
                continue
            dt = time.time() - t0
            timing[spec.name] += dt

            if importance_model and spec.name == importance_model:
                from src.ml.evaluate import importance_drops
                t1 = time.time()
                importance[wi] = importance_drops(
                    model, X_tr[:, jj], y_tr, ix_tr.get_level_values("date"), cols)
                if verbose:
                    log.info("    %-6s 重要性（训练集 %s 行 × %d 特征）%.0fs", spec.name,
                             f"{len(X_tr):,}", len(cols), time.time() - t1)

            preds[spec.name].append(pd.Series(yhat, index=ix_te))
            params[spec.name].append({"窗口": wi,
                                      "测试起": f"{sp.test[0]:%Y-%m}",
                                      **best_p, "验证损失": round(vloss, 6)})
            if verbose:
                ps = " ".join(f"{k}={v}" for k, v in best_p.items()) or "无超参"
                log.info("    %-6s %5.1fs  %s", spec.name, dt, ps)

        # 窗口之间歇一会散热。放在窗口级而不是模型级：模型之间的间隙太碎，
        # 停不出效果，反而把总时长拖长 len(specs) 倍。
        if cooldown > 0 and wi < len(splits):
            if verbose:
                log.info("    （散热 %.0fs）", cooldown)
            time.sleep(cooldown)

    out_preds = {k: pd.concat(v).sort_index() for k, v in preds.items() if v}
    return {
        "preds": out_preds,
        "y_true": pd.concat(y_parts).sort_index() if y_parts else pd.Series(dtype=float),
        "params": {k: pd.DataFrame(v) for k, v in params.items() if v},
        "cal": {k: pd.DataFrame(v) for k, v in cal_ab.items() if v},
        "timing": timing,
        "importance": importance,
        "features": feats,
        "splits": splits,
    }


def evaluate(res: dict,
             panel: pd.DataFrame,
             n_groups: int = 10,
             market: str = "A") -> pd.DataFrame:
    """把样本外预测喂进既有回测链路，汇总三类指标。

    分组数默认 **10**（GKX 与德国项目都用十分档），注意与既有单因子回测的
    5 分组**不可直接比**——十分档的多空端更极端，LS 收益天然更高，
    这是机械效应而非能力差异。要与既有因子对比时，两边都用同一个 n_groups。
    """
    from src.backtest.engine import calc_monthly_returns, group_return
    from src.backtest.metrics import calc_ic, group_summary
    from src.ml.evaluate import r2_oos, size_subsample_r2, flat_prediction_months

    y_true = res["y_true"]
    oos_months = pd.DatetimeIndex(
        y_true.index.get_level_values("date").unique()).sort_values()

    # 用样本外区间的真实持仓收益（与单因子流程同源）
    ret_m = calc_monthly_returns(market=market)
    fwd = ret_m.shift(-1).reindex(index=oos_months)

    me = panel["me"] if "me" in panel.columns else None
    rows = {}
    for name, yhat in res["preds"].items():
        common = y_true.index.intersection(yhat.index)
        yt, yp = y_true.loc[common], yhat.loc[common]

        d = pd.DataFrame({"y": yt, "yhat": yp})
        if me is not None:
            d["me"] = me.reindex(common)
        r2s = size_subsample_r2(d, "y", "yhat", me_col="me")

        w = to_wide(yp)
        cols = w.columns.intersection(fwd.columns)
        w = w.reindex(columns=cols)
        f = fwd.reindex(index=w.index, columns=cols)

        ic = calc_ic(w, f, method="spearman")
        grp = group_return(w, f, n_groups=n_groups)
        # 常数预测月份 = 模型不持仓：多空与 IC 记 0（#37，理由见 flat_prediction_months）
        flat = flat_prediction_months(w.where(f.notna()))
        if len(flat):
            ic = ic.reindex(ic.index.union(flat)).sort_index()   # calc_ic 可能不返回无法排序的月份
            ic.loc[flat] = 0.0
            fi = grp.index.intersection(flat)
            grp.loc[fi, :] = np.nan          # 无组合可言：各组收益不定义
            grp.loc[fi, "LS"] = 0.0          # 多空 = 不持仓 = 0
            log.info("  %s：%d 个月截面常数预测，按不持仓计（多空、IC 记 0）", name, len(flat))
        summ = group_summary(grp, freq=12)

        ls = summ.loc["LS"] if "LS" in summ.index else None
        rows[name] = {
            "r2_oos": r2s.get("全样本", np.nan) * 100,
            # ⚠️ 诊断列，**不是性能指标**：在测试集上拟合最优 a、b 后的 R²。
            # 它有前视偏差（oracle），唯一用途是区分两种失败：
            #   r2_oos 很负 且 r2_oracle 也很负 → 信息不够，模型真没用
            #   r2_oos 很负 但 r2_oracle 为正   → 只是预测幅度不对，校准即可修
            # 2026-08-20 就是靠这一列才没把树模型误判成「没用」：
            # LGBM 原始 −9.68%、oracle +0.81%（全场最高）。
            "r2_oracle": _r2_oracle(yt.to_numpy(), yp.to_numpy()) * 100,
            "r2_large": r2s.get("大盘(前30%)", np.nan) * 100,
            "r2_small": r2s.get("小盘(后30%)", np.nan) * 100,
            "ic": ic.mean(),
            "icir": ic.mean() / ic.std() if ic.std() else np.nan,
            "ls_ann": ls.get("年化收益", np.nan) * 100 if ls is not None else np.nan,
            "ls_sharpe": ls.get("夏普比率", np.nan) if ls is not None else np.nan,
            "ls_mdd": ls.get("最大回撤", np.nan) * 100 if ls is not None else np.nan,
            "win_rate": ls.get("月度胜率", np.nan) * 100 if ls is not None else np.nan,
            "n_obs": len(common),
        }
        res.setdefault("group_ret", {})[name] = grp
        res.setdefault("ic_series", {})[name] = ic

    df = pd.DataFrame(rows).T
    return df.sort_values("r2_oos", ascending=False)
