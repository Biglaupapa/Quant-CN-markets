# =============================================================================
# ml/run.py
# ML 资产定价主入口
#
# 使用方式（与 python -m src.main 对称）：
#   python -m src.ml.run                    # 按下方 ML_CONFIG 跑
#   python -m src.ml.run --fast             # 缩小超参网格（跑通验证用）
#   python -m src.ml.run --models OLS-H,ENet,PLS
#   python -m src.ml.run --features pool    # 只用合成池那 10 个特征
#   python -m src.ml.run --dry-run          # 只打印面板与切分方案，不训练
#
#   ── 让机器凉一点 ────────────────────────────────────────────────────────
#   python -m src.ml.run --jobs 4           # 只用 4 线程（默认已是逻辑核一半）
#   python -m src.ml.run --cooldown 10      # 每个窗口之间停 10 秒散热
#   python -m src.ml.run --jobs all         # 用满所有核（旧行为，最快也最热）
#   见 src/config/compute.py
#
# 前置条件：先跑过 `python -m src.main`，output/cache/ 里要有因子缓存。
# 本模块**只读缓存，不重算因子**。
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import FACTOR_OUTPUT_DIR

log = logging.getLogger("ml.run")


# =============================================================================
# 配置
# =============================================================================

ML_CONFIG = {
    # ── 数据 ─────────────────────────────────────────────────────────────
    "start":        "2007-01-01",   # 面板起点（与 BACKTEST_START 对齐）
    "end":          None,           # None = 到最新
    "market":       "A",
    "features":     None,           # None = 缓存里全部；也可给因子名列表
    "rank_transform": True,         # 截面 rank → [-1,1]（GKX 做法）

    # ── 样本切分（GKX 2020 附录 D 的 hybrid：训练集扩展、验证集定长前滚、每年重训）──
    # 2026-10-03 改为扩展窗口（Louis 决定，与 GKX 一致）。原文："recursively increasing the training
    # sample ... We maintain the same size of the validation sample, but roll it forward"。
    # 长度：GKX 为 18 年 / 12 年 / 1 年（美股 60 年）；A 股样本短，沿用 60 / 24 / 12 个月作为起始长度
    "train_months": 60,
    "val_months":   24,
    "test_months":  12,
    "expanding":    True,           # True = 训练集起点固定（扩展窗口，GKX）；False = 定长滚动

    # ── 训练 ─────────────────────────────────────────────────────────────
    "loss":         "huber",        # 验证集选超参的损失。**不要改成 mse**，
                                    # 理由见 models.py 顶部（收益重尾）
    "fast":         True,           # True = 缩小超参网格。首次跑通建议 True
    "calibrate":    False,          # ⛔️ 实测无效，默认关闭。详见 pipeline._fit_calibration
                                    #    的「为什么默认关闭」一节：验证段与测试段对最优
                                    #    尺度的判断**方向相反**（验证 b≈1.21 要放大，
                                    #    测试 b≈0.2~0.5 要缩小），照验证集调正好调反，
                                    #    ENet 的 R²_oos 从 +0.311 掉到 −1.164。
                                    #    代码保留，供后续尝试更好的校准方案。
    "models":       None,           # None = 全部；也可给 ["OLS-H","ENet",...]

    # ── 算力预算（散热）─────────────────────────────────────────────────
    "n_jobs":       None,           # None = 逻辑核的一半（本机 18 → 9，约慢 1.85 倍）
                                    # 给整数 = 固定线程数；"all" = 用满（最热）
    "cooldown":     0.0,            # 每个滚动窗口训练完后停几秒散热。
                                    # 限线程降的是峰值功耗，但连续跑几十分钟
                                    # 仍会积热到降频；插入间歇能把机身温度压下来，
                                    # 代价是总时长增加 n_windows × cooldown 秒。

    # ── 评估 ─────────────────────────────────────────────────────────────
    "n_groups":     10,             # GKX/JKP 惯例。与既有 5 分组结果**不可直接比**，
                                    # 故 benchmark 模块会把基准也按 10 组重算
    "cost_bps":     (0.0, 15.0, 30.0),   # 单边交易成本敏感性（基点）
    "run_benchmark":  True,         # 同口径基准对比
    "run_dm":         True,         # Diebold-Mariano 两两检验
    "run_importance": True,         # 置零法变量重要性
    "run_spanning":   True,         # 张成检验（ML 多空 vs 既有因子多空）

    "save_output":  True,
}

OUT_DIR = FACTOR_OUTPUT_DIR / "ml"


# =============================================================================
# 主流程
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ML 资产定价（Gu-Kelly-Xiu 2020 框架）")
    ap.add_argument("--fast", action="store_true", help="缩小超参网格")
    ap.add_argument("--full", action="store_true", help="完整超参网格（慢）")
    ap.add_argument("--models", default=None, help="逗号分隔的模型名")
    ap.add_argument("--features", default=None,
                    help="'pool' = 只用合成池 10 个因子；或逗号分隔的因子名")
    ap.add_argument("--n-groups", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只看面板与切分，不训练")
    ap.add_argument("--jobs", default=None,
                    help="线程预算：整数，或 'all' 用满所有核（默认 = 逻辑核一半）")
    ap.add_argument("--cooldown", type=float, default=None,
                    help="每个滚动窗口之间的散热间歇（秒），默认 0")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = dict(ML_CONFIG)
    if args.jobs is not None:
        cfg["n_jobs"] = args.jobs
    if args.cooldown is not None:
        cfg["cooldown"] = args.cooldown

    # 线程预算要在训练开始前敲定。src/__init__.py 已按默认设过一遍（那次赶在
    # numpy 导入之前，是环境变量真正生效的时机）；这里只在命令行/配置给了
    # 不同的值时改一次运行时限额（threadpoolctl 那一层）。
    from src.config.compute import apply_thread_limits
    apply_thread_limits(cfg["n_jobs"], verbose=True)
    if args.fast:
        cfg["fast"] = True
    if args.full:
        cfg["fast"] = False
    if args.n_groups:
        cfg["n_groups"] = args.n_groups
    if args.models:
        cfg["models"] = [m.strip() for m in args.models.split(",")]
    if args.features == "pool":
        from src.main import COMBINE_FACTORS
        cfg["features"] = (COMBINE_FACTORS["microstructure"]
                           + COMBINE_FACTORS["fundamental"])
    elif args.features:
        cfg["features"] = [f.strip() for f in args.features.split(",")]

    from src.ml import benchmark, pipeline
    from src.ml.cv import RollingWindowCV
    from src.ml.dataset import build_panel, feature_cols, to_wide
    from src.ml.models import default_specs, has_lightgbm

    print("=" * 68)
    print("  机器学习资产定价  (Gu, Kelly & Xiu 2020 框架)")
    print("=" * 68)

    # ── 1. 面板 ──────────────────────────────────────────────────────
    print("\n[Step 1] 构建面板...")
    panel = build_panel(factors=cfg["features"], start=cfg["start"],
                        end=cfg["end"], market=cfg["market"],
                        rank_transform=cfg["rank_transform"])
    feats = feature_cols(panel)

    # ── 2. 切分 ──────────────────────────────────────────────────────
    cv = RollingWindowCV(train_months=cfg["train_months"],
                         val_months=cfg["val_months"],
                         test_months=cfg["test_months"],
                         expanding=cfg["expanding"])
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    print("\n[Step 2] 滚动窗口切分：")
    print(cv.describe(dates).to_string(index=False))

    if args.dry_run:
        print("\n（--dry-run，未训练）")
        return 0

    # ── 3. 训练 ──────────────────────────────────────────────────────
    specs = default_specs(feats, fast=cfg["fast"])
    if cfg["models"]:
        specs = [s for s in specs if s.name in cfg["models"]]
        if not specs:
            print(f"✗ 没有匹配的模型：{cfg['models']}")
            return 1
    if not has_lightgbm():
        print("\n  ⓘ 未装 lightgbm。装了会更快更强：")
        print("    conda activate /Users/louis/MyProjects/venv")
        print("    conda install -c conda-forge lightgbm")

    print(f"\n[Step 3] 滚动训练（{len(specs)} 个模型，"
          f"{'缩小' if cfg['fast'] else '完整'}网格，{cfg['loss']} 损失）...")
    t0 = time.time()
    # 变量重要性在训练循环内、对**各窗口训练集**计算（GKX 2020 §3.3），不另行重训
    _pref = ["LGBM", "GBRT", "RF", "ENet", "PLS", "OLS-H"]
    imp_model = (next((p for p in _pref if any(sp.name == p for sp in specs)), specs[0].name)
                 if cfg["run_importance"] else None)
    res = pipeline.run(panel, specs=specs, cv=cv, loss=cfg["loss"],
                       importance_model=imp_model,
                       fast=cfg["fast"], calibrate=cfg["calibrate"],
                       cooldown=cfg["cooldown"])
    print(f"\n  训练总耗时 {time.time() - t0:.0f}s")
    print("  各模型累计耗时：" +
          "  ".join(f"{k}={v:.0f}s" for k, v in res["timing"].items() if v > 0))

    # ── 4. 评估 ──────────────────────────────────────────────────────
    print(f"\n[Step 4] 样本外评估（{cfg['n_groups']} 分组）...")
    tab = pipeline.evaluate(res, panel, n_groups=cfg["n_groups"],
                            market=cfg["market"])
    print("\n" + "─" * 68)
    print("  R²_oos 单位 %，是 GKX 口径（分母不去均值，以 0 为基准）")
    print("  参照：GKX 美股最优模型约 +0.40%，OLS 全特征约 −3.5%")
    print("  r2_oracle 是**诊断列不是性能**：测试集上最优重标定后的 R²（有前视）。")
    print("    r2_oos 负 + oracle 也负 → 信息不够；r2_oos 负 + oracle 正 → 仅尺度问题")
    print("─" * 68)
    print(tab.round(3).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if cfg["save_output"]:
        tab.to_csv(OUT_DIR / "model_summary.csv")
        for k, v in res["params"].items():
            v.to_csv(OUT_DIR / f"params_{k}.csv", index=False)
        if res.get("cal"):
            pd.concat({k: v.set_index("窗口") for k, v in res["cal"].items()},
                      names=["模型"]).to_csv(OUT_DIR / "calibration.csv")

    # ── 5. 同口径基准 ────────────────────────────────────────────────
    if cfg["run_benchmark"]:
        print(f"\n[Step 5] 同口径基准对比（含换手率与成本敏感性）...")
        cmp = benchmark.compare(res, panel, n_groups=cfg["n_groups"],
                                market=cfg["market"])
        print("\n" + cmp.round(3).to_string())
        if cfg["save_output"]:
            cmp.to_csv(OUT_DIR / "benchmark_compare.csv")

    # ── 6. Diebold-Mariano ───────────────────────────────────────────
    if cfg["run_dm"] and len(res["preds"]) > 1:
        from src.ml.evaluate import dm_matrix
        print("\n[Step 6] Diebold-Mariano 两两检验")
        print("  正值 = 列模型优于行模型；*/**/*** = 10%/5%/1% 显著")
        common = None
        for v in res["preds"].values():
            common = v.index if common is None else common.intersection(v.index)
        preds = {k: v.loc[common].to_numpy() for k, v in res["preds"].items()}
        dm = dm_matrix(preds, res["y_true"].loc[common].to_numpy())
        print("\n" + dm.to_string())
        if cfg["save_output"]:
            dm.to_csv(OUT_DIR / "dm_matrix.csv")

    # ── 7. 变量重要性 ────────────────────────────────────────────────
    if cfg["run_importance"] and res.get("importance"):
        print(f"\n[Step 7] 变量重要性（置零法，{imp_model}，各窗口训练集，跨 "
              f"{len(res['importance'])} 个窗口平均；GKX 2020 §3.3）...")
        try:
            imp, diag = _importance_summary(res["importance"])
            print("\n  前 15 名（%，r2 = GKX 口径；ic = 补充口径，无直接文献依据）：")
            print(imp.head(15).mul(100).round(2).to_string())
            print("\n  检验：")
            for k, v in diag.items():
                print(f"    {k}: {v}")
            if cfg["save_output"]:
                imp.to_csv(OUT_DIR / "variable_importance.csv")
                pd.concat(res["importance"], names=["窗口", "特征"]).to_csv(
                    OUT_DIR / "variable_importance_by_window.csv")
                pd.Series(diag).to_csv(OUT_DIR / "variable_importance_diag.csv")
        except Exception as e:
            log.warning("  变量重要性失败：%s", e)

    # ── 8. 张成检验 ──────────────────────────────────────────────────
    if cfg["run_spanning"] and "group_ret" in res:
        print("\n[Step 8] 张成检验：ML 多空收益 vs 既有因子多空收益")
        print("  α 显著为正 → ML 捕捉到既有因子张不成的信息")
        try:
            sp_tab = _spanning(res, cfg)
            if cfg["save_output"] and sp_tab is not None:
                # attrs 在 to_csv 往返中会丢，故把 α年化 / R² / 样本数
                # 作为额外几行写进表里，保证落盘的文件是自足的
                meta = pd.DataFrame(
                    {"系数": [sp_tab.attrs["alpha_ann"], sp_tab.attrs["r2"],
                              sp_tab.attrs["n"]]},
                    index=["_α年化", "_R²", "_样本月数"])
                pd.concat([sp_tab, meta]).to_csv(OUT_DIR / "spanning_test.csv")
        except Exception as e:
            log.warning("  张成检验失败：%s", e)

    print(f"\n结果已保存至 {OUT_DIR}")
    return 0


# -----------------------------------------------------------------------------

def _importance_summary(by_win: dict) -> tuple[pd.DataFrame, dict]:
    """
    跨窗口汇总置零法重要性。

    GKX（2020）§3.3：在每个训练样本上算 R² 下降，再「average these into a single
    importance measure」，归一化到和为 1。这里先跨窗口平均原始下降量，再归一化
    （负下降记 0）。IC 口径同法处理，作补充。

    两项检验（2026-10-03 Louis 同意）：
      - 跨窗口稳定性：各窗口重要性排名两两 Spearman 的均值（越高越稳）
      - 口径一致性：两种口径汇总排名的 Spearman，及前 20 名重合数
    """
    from src.ml.evaluate import normalize_importance
    from itertools import combinations
    raw = pd.concat(by_win, names=["窗口", "特征"])
    mean = raw.groupby(level="特征").mean()
    imp = pd.DataFrame({m: normalize_importance(mean[m]) for m in ("r2", "ic")})
    imp = imp.sort_values("r2", ascending=False)

    def stability(m):
        w = raw[m].unstack("特征")
        cs = [w.iloc[i].corr(w.iloc[j], method="spearman")
              for i, j in combinations(range(len(w)), 2)]
        return round(float(np.nanmean(cs)), 3) if cs else float("nan")

    diag = {
        "窗口数": len(by_win),
        "跨窗口稳定性_r2（两两 Spearman 均值）": stability("r2"),
        "跨窗口稳定性_ic（两两 Spearman 均值）": stability("ic"),
        "口径一致性（r2 vs ic 汇总排名 Spearman）": round(float(imp.r2.corr(imp.ic, method="spearman")), 3),
        "前20名重合数": len(set(imp.r2.nlargest(20).index) & set(imp.ic.nlargest(20).index)),
    }
    try:
        from src.factors.accounting import ACCOUNTING_FACTORS
        acc = imp.index.isin(list(ACCOUNTING_FACTORS))
        diag["会计块合计占比_r2"] = round(float(imp.r2[acc].sum()), 3)
        diag["会计块合计占比_ic"] = round(float(imp.ic[acc].sum()), 3)
        diag["会计块特征数"] = int(acc.sum())
    except Exception:
        pass
    return imp, diag


def _importance_last_window(panel, specs, cv, feats, cfg):
    """[已弃用 2026-10-03] 旧做法：最后一个满窗、在**测试集**上算——与 GKX（训练集、跨窗口平均）不符。"""
    """在最后一个窗口上重训最强的树模型，算置零法重要性。

    只用一个窗口是成本考虑：置零法要对每个特征各跑一次预测，
    61 个特征 × 13 个窗口 = 793 次，没有必要。
    """
    from src.ml.evaluate import variable_importance
    from src.ml.models import fit_with_validation
    from src.ml.pipeline import _slice

    pref = ["LGBM", "GBRT", "RF", "ENet", "PLS", "OLS-H"]
    spec = next((s for p in pref for s in specs if s.name == p), specs[0])

    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    splits = list(cv.split(dates))

    # 取**最后一个满窗**，不是最后一个窗口。末窗的测试段常常不足
    # test_months 个月（数据到头了，实测只有 7 个月），在这么薄的样本上
    # 算置零法重要性，排名会被单季度行情带偏。
    full = [s for s in splits if len(s.test) >= cv.test_months]
    sp = full[-1] if full else splits[-1]

    X_tr, y_tr, _ = _slice(panel, sp.train, feats)
    X_va, y_va, _ = _slice(panel, sp.val, feats)
    X_te, y_te, ix = _slice(panel, sp.test, feats)

    model, _, _ = fit_with_validation(spec, X_tr, y_tr, X_va, y_va, loss=cfg["loss"])
    Xdf = pd.DataFrame(X_te, columns=feats, index=ix)
    print(f"  模型：{spec.name}，窗口 {sp.test[0]:%Y-%m}~{sp.test[-1]:%Y-%m}"
          f"（{len(sp.test)} 个测试月，{len(X_te):,} 个观测）")
    return variable_importance(model, Xdf, y_te, feats)


def _spanning(res, cfg):
    """ML 多空收益对既有因子多空收益回归。"""
    from src.ml import benchmark
    from src.ml.evaluate import spanning_test
    from src.backtest.engine import calc_monthly_returns, group_return

    fwd = calc_monthly_returns().shift(-1)
    months = pd.DatetimeIndex(
        res["y_true"].index.get_level_values("date").unique()).sort_values()

    # 解释变量：合成因子 + 两个最强单因子的多空收益
    rhs = {}
    for label, score in [("composite", benchmark.build_composite()),
                         ("size", benchmark.load_factor("size")),
                         ("amihud", benchmark.load_factor("amihud"))]:
        s = score.reindex(index=months).dropna(how="all")
        cols = s.columns.intersection(fwd.columns)
        g = group_return(s.reindex(columns=cols),
                         fwd.reindex(index=s.index, columns=cols),
                         n_groups=cfg["n_groups"])
        rhs[label] = g["LS"]
    X = pd.DataFrame(rhs)

    best = max(res["group_ret"], key=lambda k: res["group_ret"][k]["LS"].mean())
    y = res["group_ret"][best]["LS"]
    out = spanning_test(y, X)
    print(f"\n  被解释变量：{best} 的多空收益（{out.attrs['n']} 个月）")
    print(out.round(4).to_string())
    print(f"\n  α 年化 = {out.attrs['alpha_ann'] * 100:.2f}%   R² = {out.attrs['r2']:.3f}")
    a = out.loc["const"]
    verdict = ("✅ α 显著为正 → ML 有既有因子张不成的增量"
               if a["p值"] < 0.05 and a["系数"] > 0
               else "⚠️ α 不显著 → ML 的收益基本能被既有因子解释")
    print(f"  {verdict}")
    return out


if __name__ == "__main__":
    sys.exit(main())
