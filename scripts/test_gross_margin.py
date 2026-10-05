"""毛利率因子快速检验：运行时临时注册，不改任何源文件、不写因子缓存。

2026-09-29 结果（2007-01~2026-08，5 分组）：
    gmar_raw  IC 0.003  ICIR 0.03  LS -0.4%
    gmar      IC 0.007  ICIR 0.07  LS  1.3%  （市值+行业中性化）
    gp_at     IC 0.011  ICIR 0.09  LS  1.8%
    gmar vs gp_at 截面秩相关 0.462；分组均非单调
数据：财汇 PIT（BIZINCO / BIZCOST），经 src/factors/accounting.py 的 _ttm。
用法：cd ~/MyProjects/Quant && python3 scripts/test_gross_margin.py
"""
import sys; sys.path.insert(0, "/Users/louis/MyProjects/Quant")
import numpy as np, pandas as pd
import src.factors.accounting as acc
from src.main import BACKTEST_CONFIG
from src.backtest.engine import calc_monthly_returns, group_return
from src.backtest.metrics import calc_ic, group_summary, ic_summary

gm = lambda: ((acc._ttm("BIZINCO") - acc._ttm("BIZCOST")) / acc._ttm("BIZINCO"))
acc.SPECS["gmar_raw"] = (gm, False, "毛利率（不中性化）")
acc.SPECS["gmar"]     = (gm, True,  "毛利率（市值+行业中性化）")
cands = {"gmar_raw": acc._make("gmar_raw"), "gmar": acc._make("gmar"), "gp_at": acc._make("gp_at")}

start, end = BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"]
fwd = calc_monthly_returns(start=start, end=end, market="A").shift(-1)
rows, F = {}, {}
for n, f in cands.items():
    x = f(start, end); F[n] = x
    ic = calc_ic(x, fwd, method="spearman"); st = ic_summary(ic)
    s = group_summary(group_return(x, fwd, n_groups=5), freq=12)
    cov = x.notna().sum(axis=1); cov = cov[cov > 0]
    rows[n] = {"首月": cov.index.min().strftime("%Y-%m"), "月数": int(ic.notna().sum()), "截面均值只数": int(cov.mean()),
               "IC均值": st["IC均值"], "ICIR": st["ICIR"], "IC>0占比": st["IC>0占比"],
               "LS年化": s.loc["LS","年化收益"], "LS夏普": s.loc["LS","夏普比率"], "LS回撤": s.loc["LS","最大回撤"],
               "G1年化": s.loc["G1","年化收益"], "G5年化": s.loc["G5","年化收益"]}
    g = s["年化收益"].drop("LS"); rows[n]["分组单调"] = bool(g.is_monotonic_increasing or g.is_monotonic_decreasing)
pd.set_option("display.width", 220)
print(f"\n{start} ~ {end}，5 分组（G5 = 因子值最高组）\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))
a, b = F["gmar"].align(F["gp_at"], join="inner")
print(f"\n截面秩相关 gmar vs gp_at：{a.T.rank().corrwith(b.T.rank()).mean():.3f}")
# 分时段稳定性
ic = calc_ic(F["gmar"], fwd, method="spearman").dropna()
bins = pd.cut(ic.index.year, [2006, 2012, 2017, 2021, 2026], labels=["2007-12", "2013-17", "2018-21", "2022-26"])
print("gmar 分时段 IC 均值:", ic.groupby(bins, observed=True).mean().round(4).to_dict())
