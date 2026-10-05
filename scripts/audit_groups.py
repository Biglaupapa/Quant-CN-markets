"""
分组体检（2026-10-05）：组建日股票池内，逐因子逐月检查 5 分组是否完整、均衡。
    python scripts/audit_groups.py            # g 关（默认）
    python scripts/audit_groups.py --g on
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src.config.settings import FACTOR_OUTPUT_DIR
from src.backtest.engine import calc_monthly_returns
from src.strategy.optimizer import group_by_score
from src.main import FACTOR_FLAGS

ap = argparse.ArgumentParser(); ap.add_argument("--g", choices=["on", "off"], default="off")
args = ap.parse_args()
cfg = {"exclude_bottom_size": args.g == "on"}
fwd = calc_monthly_returns("2007-01-01", "2026-09-30", formation_config=cfg).shift(-1)
months = fwd.index[fwd.notna().any(axis=1)]

rows = []
for name, on in FACTOR_FLAGS.items():
    p = FACTOR_OUTPUT_DIR / "cache" / f"{name}.csv"
    if not on or not p.exists():
        continue
    f = pd.read_csv(p, index_col=0); f.index = pd.to_datetime(f.index)
    f = f.reindex(index=months, columns=fwd.columns).where(fwd.loc[months].notna())
    has = f.notna().sum(axis=1)
    g = group_by_score(f[has > 0], 5)
    sizes = pd.DataFrame({k: (g == k).sum(axis=1) for k in range(1, 6)})
    grouped = sizes.sum(axis=1) > 0
    s = sizes[grouped]
    empty_any = (s == 0).any(axis=1)
    rows.append({
        "因子": name, "有值月": int((has > 0).sum()), "分组月": int(grouped.sum()),
        "跳过(<25只)": int(((has > 0) & ~grouped.reindex(has.index, fill_value=False)).sum()),
        "有空组月": int(empty_any.sum()),
        "G1空": int((s[1] == 0).sum()), "G5空": int((s[5] == 0).sum()),
        "组人数比 max/min 中位": round(float((s.max(axis=1) / s.replace(0, np.nan).min(axis=1)).median()), 3),
        "组人数比 max/min 最大": round(float((s.max(axis=1) / s.replace(0, np.nan).min(axis=1)).max()), 2),
        "每组中位只数": int(s.median(axis=1).median()) if len(s) else 0,
    })
t = pd.DataFrame(rows).set_index("因子")
out = FACTOR_OUTPUT_DIR / "checks"; out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / f"audit_groups_g{args.g}_20261005.csv")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
print(f"因子数 {len(t)}；有空组的因子 {(t['有空组月'] > 0).sum()}；有跳过月的因子 {(t['跳过(<25只)'] > 0).sum()}")
flag = t[(t["有空组月"] > 0) | (t["跳过(<25只)"] > 0) | (t["组人数比 max/min 最大"] > 1.5)]
print(flag.to_string())
print("\n全体：组人数比中位数的分布", t["组人数比 max/min 中位"].describe().round(3).to_dict())
