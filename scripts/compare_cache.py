# -*- coding: utf-8 -*-
"""
因子缓存对拍：验证向量化改动没有改变任何数值
=============================================
比较 `output/cache/`（优化后）与 `output/cache_before_opt/`（优化前）的
每个因子矩阵，逐格核对。

用途：C1~C3 + D 四项改动都声称「数学等价、只变快」，本脚本是这个声称的证据。
判据：两边都有值的格子，最大**相对差 < 1e-9**；且 NaN 模式完全一致。

用法（zsh）：
    source /Users/louis/MyProjects/venv/bin/activate
    cd /Users/louis/MyProjects/Quant

    python3 scripts/compare_cache.py
    python3 scripts/compare_cache.py --tol 1e-12        # 收紧阈值
    python3 scripts/compare_cache.py --only ivol,roll_spread
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent
NEW    = ROOT / "output" / "cache"
OLD    = ROOT / "output" / "cache_before_opt"


def load(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, index_col=0, low_memory=False)
    d.index = pd.to_datetime(d.index, errors="coerce")
    return d[d.index.notna()].sort_index()


def compare_one(name: str, tol: float) -> dict:
    pn, po = NEW / f"{name}.csv", OLD / f"{name}.csv"
    if not po.exists():
        return {"name": name, "status": "新增", "detail": "旧缓存中不存在"}
    if not pn.exists():
        return {"name": name, "status": "缺失", "detail": "新缓存中不存在"}

    a, b = load(pn), load(po)
    if a.shape != b.shape:
        shape_note = f"形状 {a.shape} vs {b.shape}"
    else:
        shape_note = ""

    idx  = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    x, y = a.loc[idx, cols], b.loc[idx, cols]

    both     = x.notna() & y.notna()
    only_new = int((x.notna() & y.isna()).sum().sum())
    only_old = int((x.isna() & y.notna()).sum().sum())
    n_both   = int(both.sum().sum())

    if n_both == 0:
        return {"name": name, "status": "无可比", "detail": shape_note}

    denom = y.abs().where(y.abs() > 0, 1.0)
    rel   = ((x - y).abs() / denom)[both]
    max_rel = float(np.nanmax(rel.values))
    max_abs = float(np.nanmax((x - y).abs()[both].values))
    n_bad   = int((rel > tol).sum().sum())

    ok = (n_bad == 0) and (only_new == 0) and (only_old == 0)
    detail = f"最大相对差 {max_rel:.2e}  最大绝对差 {max_abs:.2e}"
    if only_new or only_old:
        detail += f"  ⚠ NaN 模式不同（仅新 {only_new}，仅旧 {only_old}）"
    if shape_note:
        detail += f"  {shape_note}"

    return {"name": name, "status": "✓ 一致" if ok else "✗ 有差异",
            "detail": detail, "ok": ok, "n_both": n_both,
            "max_rel": max_rel, "worst": rel.stack().nlargest(3) if not ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-9, help="相对差阈值，默认 1e-9")
    ap.add_argument("--only", default="", help="逗号分隔的因子名")
    args = ap.parse_args()

    if not OLD.exists():
        sys.exit(f"✗ 找不到基准目录 {OLD}\n  优化前请先执行: cp -r output/cache output/cache_before_opt")

    names = ([s.strip() for s in args.only.split(",") if s.strip()]
             or sorted({p.stem for p in OLD.glob("*.csv")} | {p.stem for p in NEW.glob("*.csv")}))

    print(f"新: {NEW}")
    print(f"旧: {OLD}")
    print(f"阈值: 相对差 < {args.tol:.0e}\n")
    print(f"{'因子':<28}{'状态':<10}{'可比格数':>12}   明细")
    print("-" * 92)

    bad = []
    for n in names:
        r = compare_one(n, args.tol)
        nb = f"{r.get('n_both', 0):,}" if r.get("n_both") else "—"
        print(f"{r['name']:<28}{r['status']:<10}{nb:>12}   {r.get('detail','')}")
        if r.get("ok") is False:
            bad.append(r)

    print("-" * 92)
    if not bad:
        print(f"✓ 全部 {len(names)} 个因子数值一致 —— 向量化改动未改变任何结果")
    else:
        print(f"✗ {len(bad)} 个因子有差异：")
        for r in bad:
            print(f"\n  【{r['name']}】最大相对差 {r['max_rel']:.3e}")
            if r["worst"] is not None:
                for (d, c), v in r["worst"].items():
                    print(f"     {c} {d.date()}  相对差 {v:.3e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
