"""换手因子四版本对比：流通 vs 自由流通 × 中性化与否。复用 Quant 主流程的同一套函数。

2026-09-29 结果见 Database/docs/【核验】聚源自由流通换手率.md §四。
用法：python3 scripts/compare_turnover_ff.py [要强制重算的因子名 ...]
"""
import sys
sys.path.insert(0, "/Users/louis/MyProjects/Quant")
import pandas as pd

from src.main import _get_factor_func, _load_factor_cached, BACKTEST_CONFIG, _as_list
from src.backtest.engine import calc_monthly_returns, group_return
from src.backtest.metrics import calc_ic, group_summary, ic_summary

NAMES = ["turnover_20", "turnover_20_ff", "turnover_20_neutral", "turnover_20_ff_neutral"]
start, end = BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"]
ng = _as_list(BACKTEST_CONFIG["n_groups"])[0]
force = set(sys.argv[1:])          # 传入因子名则强制重算该因子

ret = calc_monthly_returns(start=start, end=end, market="A")
fwd = ret.shift(-1)
rows, facs = {}, {}
for n in NAMES:
    f = _load_factor_cached(n, _get_factor_func(n), start, end, n in force, cache_prefix="")
    facs[n] = f
    ic = calc_ic(f, fwd, method="spearman")
    s = group_summary(group_return(f, fwd, n_groups=ng), freq=12)
    st = ic_summary(ic)
    rows[n] = {"月数": int(ic.notna().sum()), "IC均值": st["IC均值"], "ICIR": st["ICIR"],
               "LS年化": s.loc["LS", "年化收益"], "LS夏普": s.loc["LS", "夏普比率"],
               "LS回撤": s.loc["LS", "最大回撤"], "LS胜率": s.loc["LS", "月度胜率"],
               "G1年化": s.iloc[0]["年化收益"], f"G{ng}年化": s.loc[f"G{ng}", "年化收益"]}

pd.set_option("display.width", 200)
print(f"\n区间 {start} ~ {end}，{ng} 分组\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))

# 两个口径的截面相关（月度 Spearman 均值）：越低说明自由流通口径带来的新信息越多
for a, b in [("turnover_20", "turnover_20_ff"), ("turnover_20_neutral", "turnover_20_ff_neutral")]:
    fa, fb = facs[a].align(facs[b], join="inner")
    c = fa.T.rank().corrwith(fb.T.rank()).dropna()
    print(f"截面秩相关 {a} vs {b}：均值 {c.mean():.4f}，最低 {c.min():.4f}（{c.idxmin():%Y-%m}）")
