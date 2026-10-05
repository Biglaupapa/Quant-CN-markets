"""
待办 #23 诊断：OLS-H 在 112 特征下 R²_oos 失控（约 −40%），排序却正常。

逐窗口按 pipeline 原样重拟合 OLS-H（train+val 合并、HuberRegressor ε=1.35、max_iter=500、alpha=1e-4），记录：
  收敛（n_iter_ 是否触顶）、scale_、系数范数、测试集预测标准差 / 真实标准差、单窗 R²_oos、Pearson 相关；
对照：普通 OLS（LinearRegression）同样本。主框架设置（g 关）。

    python scripts/diag_olsh.py [--features base61]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config.settings import FACTOR_OUTPUT_DIR
from src.ml.cv import RollingWindowCV
from src.ml.dataset import build_panel, feature_cols
from src.ml.evaluate import r2_oos
from src.ml.models import _ols_huber
from src.ml.pipeline import _slice
from src.ml.run import ML_CONFIG as CONFIG

ap = argparse.ArgumentParser()
ap.add_argument("--features", default=None, help="base61 = output/ml/base61_features.txt")
args = ap.parse_args()

feats_arg = None
if args.features == "base61":
    feats_arg = (FACTOR_OUTPUT_DIR / "ml" / "base61_features.txt").read_text().strip().split(",")

panel = build_panel(factors=feats_arg, start=CONFIG["start"], end=CONFIG["end"],
                    market=CONFIG["market"], rank_transform=CONFIG["rank_transform"])
feats = feature_cols(panel)
dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
cv = RollingWindowCV(train_months=CONFIG["train_months"], val_months=CONFIG["val_months"],
                     test_months=CONFIG["test_months"], expanding=CONFIG["expanding"])

rows, y_all, p_all, o_all = [], [], [], []
for wi, sp in enumerate(cv.split(dates), 1):
    X_tr, y_tr, _ = _slice(panel, sp.train, feats)
    X_va, y_va, _ = _slice(panel, sp.val, feats)
    X_te, y_te, _ = _slice(panel, sp.test, feats)
    if len(X_te) == 0:
        continue
    Xf, yf = np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va])

    t0 = time.time()
    m = _ols_huber()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(Xf, yf)
        conv_warn = any("converge" in str(x.message).lower() for x in w)
    th = time.time() - t0
    p = m.predict(X_te)
    o = LinearRegression().fit(Xf, yf).predict(X_te)

    rows.append({
        "窗口": wi, "测试起": f"{sp.test[0]:%Y-%m}", "n_train": len(yf),
        "n_iter": m.n_iter_, "未收敛": conv_warn or m.n_iter_ >= m.max_iter, "秒": round(th, 1),
        "scale_": m.scale_, "|coef|": np.linalg.norm(m.coef_), "截距": m.intercept_,
        "预测std/真实std": p.std() / y_te.std(), "预测均值": p.mean(), "真实均值": y_te.mean(),
        "R2_huber%": 100 * r2_oos(y_te, p), "R2_ols%": 100 * r2_oos(y_te, o),
        "corr_huber": np.corrcoef(p, y_te)[0, 1], "corr_ols": np.corrcoef(o, y_te)[0, 1],
    })
    y_all.append(y_te); p_all.append(p); o_all.append(o)
    print(pd.Series(rows[-1]).to_dict(), flush=True)

tab = pd.DataFrame(rows)
y, p, o = map(np.concatenate, (y_all, p_all, o_all))
print(f"\n合计 R²_oos：Huber {100 * r2_oos(y, p):.3f}%   OLS {100 * r2_oos(y, o):.3f}%   （特征 {len(feats)}）")
out = FACTOR_OUTPUT_DIR / "checks"
out.mkdir(parents=True, exist_ok=True)
tab.to_csv(out / f"diag_olsh_{len(feats)}.csv", index=False)
pd.set_option("display.width", 250)
print(tab.round(4).to_string(index=False))
