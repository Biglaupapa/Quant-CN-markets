"""
ML 组重要性（GKX 四大类，第 4 类拆估值 / 会计）—— 扩展窗口、完整网格，与 src.ml.run 同一设置。

每个窗口：训练 → 在**训练集**上把一组特征同时置零 → 记 R² 与 IC 的下降（GKX 2020 §3.3：
within each training sample，再跨样本平均）→ 跨窗口平均 → 归一化（负值记 0）。
模型：LGBM、OLS-H。分组：docs/reference/feature_groups_gkx.csv（Louis 2026-10-03 确认方案 A）。
输出：output/ml/group_importance{,_by_window}.csv
用法：python scripts/ml_group_importance.py [--g off|on] [--out output/ml/group_importance_xxx]
  --g   规则 g（剔 A 股市值最小 30%）开 / 关，默认关（主框架默认）
  --out 输出目录，默认 output/ml/（文件名 group_importance{,_by_window}.csv）
"""
import argparse, sys; sys.path.insert(0, "/Users/louis/MyProjects/Quant")
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--g", choices=["off", "on"], default="off")
ap.add_argument("--out", default="output/ml"); args = ap.parse_args()
from src.config import settings
settings.FORMATION_CONFIG["exclude_bottom_size"] = (args.g == "on")   # 标签与评估股票池随之改变
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
from src.ml.run import ML_CONFIG
from src.ml.dataset import build_panel, feature_cols
from src.ml.cv import RollingWindowCV
from src.ml.models import default_specs, fit_with_validation
from src.ml.pipeline import _slice, mask_immature_features
from src.ml.evaluate import importance_drops_groups, normalize_importance
from src.config.compute import apply_thread_limits

cfg = dict(ML_CONFIG); apply_thread_limits(9)
G = pd.read_csv("docs/reference/feature_groups_gkx.csv")
G["组"] = np.where(G["GKX类别"].str.startswith("4"), np.where(G["会计块"], "4b 会计", "4a 估值"), G["GKX类别"])
panel = build_panel(factors=cfg["features"], start=cfg["start"], end=cfg["end"], market=cfg["market"],
                    rank_transform=cfg["rank_transform"])
feats = feature_cols(panel)
groups = {g: [f for f in m if f in feats] for g, m in G.groupby("组")["特征"].apply(list).items()}
print("特征", len(feats), "组：", {g: len(m) for g, m in groups.items()})
cv = RollingWindowCV(train_months=cfg["train_months"], val_months=cfg["val_months"],
                     test_months=cfg["test_months"], expanding=cfg["expanding"])
dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
specs = [s for s in default_specs(feats, fast=False) if s.name in ("LGBM", "OLS-H")]
rows = []
for wi, sp in enumerate(cv.split(dates), 1):
    Xtr, ytr, ixtr = _slice(panel, sp.train, feats); Xva, yva, ixva = _slice(panel, sp.val, feats)
    _, (Xtr, Xva) = mask_immature_features(np.vstack([Xtr, Xva]),
                                           ixtr.get_level_values("date").append(ixva.get_level_values("date")),
                                           feats, cfg["min_feature_months"], Xtr, Xva)   # 与 pipeline 一致（#23）
    for spec in specs:
        m, _, _ = fit_with_validation(spec, Xtr, ytr, Xva, yva, loss=cfg["loss"])
        d = importance_drops_groups(m, Xtr, ytr, ixtr.get_level_values("date"), feats, groups)
        d["窗口"], d["模型"] = wi, spec.name
        rows.append(d.rename_axis("组").reset_index())
    print(f"窗口 {wi} 完成", flush=True)
by = pd.concat(rows); by.to_csv(OUT / "group_importance_by_window.csv", index=False)
out = []
for mdl, g in by.groupby("模型"):
    mean = g.groupby("组")[["r2", "ic"]].mean()
    t = pd.DataFrame({"r2": normalize_importance(mean.r2), "ic": normalize_importance(mean.ic)})
    t["特征数"] = g.groupby("组")["n"].first(); t["模型"] = mdl
    t["r2为正的窗口数"] = g.groupby("组")["r2"].apply(lambda s: int((s > 0).sum()))
    t["ic为正的窗口数"] = g.groupby("组")["ic"].apply(lambda s: int((s > 0).sum()))
    out.append(t.rename_axis("组").reset_index())
res = pd.concat(out); res.to_csv(OUT / "group_importance.csv", index=False)
pd.set_option("display.width", 200)
print(res.assign(r2=lambda x: (x.r2 * 100).round(1), ic=lambda x: (x.ic * 100).round(1)).to_string(index=False))
