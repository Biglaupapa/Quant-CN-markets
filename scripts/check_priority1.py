"""
第一优先三项口径检查（2026-10-05）：同一份数据上「现行 vs 修正」对比，主框架设置（g 关）。

  #15  rolling 未传 min_periods：MIN_ROLLING_VALID_DAYS 不生效（turnover_20 族、volatility_30）
  #34  合成打分的 winsorize / z-score 在因子全截面上算，而非组建日股票池
  #21  ps_liq_beta 的时序回归用了组建日筛选后的收益

    python scripts/check_priority1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.engine import calc_monthly_returns, group_return
from src.backtest.metrics import calc_ic, group_summary
from src.config.settings import MIN_ROLLING_VALID_DAYS, FACTOR_OUTPUT_DIR
from src.factors import microstructure as ms
from src.factors.base import winsorize, standardize

START, END = "2007-01-01", "2026-09-30"
OUT = FACTOR_OUTPUT_DIR / "checks"
OUT.mkdir(parents=True, exist_ok=True)

ret_m = calc_monthly_returns(start=START, end=END)
fwd = ret_m.shift(-1)


def evaluate(f: pd.DataFrame, sign: int = 1) -> dict:
    f = f.loc[START:END]
    ic = calc_ic(f, fwd, method="spearman").dropna()
    s = group_summary(group_return(f, fwd, n_groups=5), freq=12).loc["LS"]
    pool = f.reindex(index=fwd.index, columns=fwd.columns).where(fwd.notna())
    cov = (pool.notna().sum(axis=1) / fwd.notna().sum(axis=1)).loc[START:END]
    return {"IC": ic.mean(), "ICIR": ic.mean() / ic.std(),
            "LS年化": sign * s["年化收益"], "LS夏普": sign * s["夏普比率"],
            "池内覆盖": cov[cov.index <= fwd.dropna(how="all").index.max()].mean()}


def xs_rankcorr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    a, b = a.align(b, join="inner")
    return float(np.nanmean([a.loc[d].corr(b.loc[d], method="spearman")
                             for d in a.index[::6] if a.loc[d].notna().sum() > 100]))


rows = {}

# ── #15 ─────────────────────────────────────────────────────────────────
_orig_rolling = pd.DataFrame.rolling


def _rolling_with_min(self, window, *a, **k):
    if "min_periods" not in k and not a and window in (ms.TURNOVER_WINDOW, ms.VOLATILITY_WINDOW):
        k["min_periods"] = MIN_ROLLING_VALID_DAYS
    return _orig_rolling(self, window, *a, **k)


for name, sign in [("turnover_20", -1), ("turnover_20_neutral", -1),
                   ("turnover_20_ff", -1), ("turnover_20_ff_neutral", -1),
                   ("volatility_30", -1)]:
    fn = getattr(ms, f"calc_{name}")
    pd.DataFrame.rolling = _orig_rolling
    old = fn(START, END)
    pd.DataFrame.rolling = _rolling_with_min
    new = fn(START, END)
    pd.DataFrame.rolling = _orig_rolling
    rows[(f"#15 {name}", "现行")] = evaluate(old, sign)
    rows[(f"#15 {name}", "修正")] = evaluate(new, sign)
    rows[(f"#15 {name}", "修正")]["截面秩相关"] = xs_rankcorr(old, new)
    print(f"[#15] {name} 完成", flush=True)

# ── #34：合成在组建日股票池内重新去极值 + 标准化 ──────────────────────
cache = FACTOR_OUTPUT_DIR / "cache"
from src.main import COMBINE_FACTORS, FACTOR_DIRECTIONS
from src.strategy.combine_factors import align_factor_directions, combine_factors

names = COMBINE_FACTORS["microstructure"] + COMBINE_FACTORS["fundamental"]
fac = {}
for n in names:
    df = pd.read_csv(cache / f"{n}.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    fac[n] = df
dirs = {n: FACTOR_DIRECTIONS[n] for n in names}
comp_old = combine_factors(align_factor_directions(fac, dirs), weights="equal")

inpool = {n: standardize(winsorize(df.reindex(index=fwd.index, columns=fwd.columns).where(fwd.notna())))
          for n, df in fac.items()}
comp_new = combine_factors(align_factor_directions(inpool, dirs), weights="equal")
rows[("#34 合成等权", "现行")] = evaluate(comp_old)
rows[("#34 合成等权", "修正")] = evaluate(comp_new)
rows[("#34 合成等权", "修正")]["截面秩相关"] = xs_rankcorr(comp_old.where(fwd.notna()), comp_new)
print("[#34] 完成", flush=True)

# ── #21：ps_liq_beta 用全部可得收益 ───────────────────────────────────
import src.backtest.engine as eng
from src.data.loader import load_data, to_monthly

old = ms._calc_ps_liq_beta(START, END)
close = load_data([ms._CLOSE], start=START, end=END)[ms._CLOSE]
raw_ret = to_monthly(close, method="last").pct_change(fill_method=None)
_orig_cmr = eng.calc_monthly_returns
eng.calc_monthly_returns = lambda *a, **k: raw_ret
try:
    new = ms._calc_ps_liq_beta(START, END)
finally:
    eng.calc_monthly_returns = _orig_cmr
rows[("#21 ps_liq_beta", "现行")] = evaluate(old)
rows[("#21 ps_liq_beta", "修正")] = evaluate(new)
rows[("#21 ps_liq_beta", "修正")]["截面秩相关"] = xs_rankcorr(old, new)
print("[#21] 完成", flush=True)

tab = pd.DataFrame(rows).T
tab.to_csv(OUT / "priority1_20261005.csv")
pd.set_option("display.width", 200)
print(tab.round(4).to_string())
