"""
复现 Liu, Stambaugh, Yuan (2019, JFE) 的 CH-3 因子（Table 3）与异象多空收益（Table 6 Panel A），
检验本框架新口径（组建日规则 a–g + 收盘到收盘）的结果量级是否与文献可比。

方法（按 LSY 原文；与原文的差异在 docs/【方法】回测口径与文献对齐.md §十一 列明）：
- 股票池：settings.FORMATION_CONFIG 全套，并显式打开 g（剔总市值最小 30%；主框架默认关闭）
- 市值：t 月末 a_market_value（财汇 TOTMKTCAP，A 股市值含限售股，与 LSY 一致），用于加权与 size
- 换手：turn（框架权威来源，VOLUME / LIQSHARE）。LSY 分母为总股本，口径差异见方法文档 §十一
- CH-3：剩余股票按市值中位数分 S/B，按 EP 分 30/40/30（V/M/G），6 个市值加权组合；
        SMB = (S/V+S/M+S/G)/3 − (B/V+B/M+B/G)/3；VMG = (S/V+B/V)/2 − (S/G+B/G)/2；
        MKT = 股票池市值加权收益 − 一年期存款利率/12
- 异象：十分组、市值加权，多空 = 预期高收益一端 − 低收益一端（月度 %）

用法：cd ~/MyProjects/Quant && python3 scripts/replicate_lsy.py
"""
import sys
sys.path.insert(0, "/Users/louis/MyProjects/Quant")
import numpy as np
import pandas as pd
from src.data.loader import load_data
from src.data.universe import build_formation_mask, _month_end_rows
from src.backtest.engine import calc_monthly_returns

S, E = "2006-01-01", "2026-09-30"
LSY_T3 = {"MKT": 0.66, "SMB": 1.03, "VMG": 1.14}
LSY_T6 = {"size（小−大）": 1.09, "EP（高−低）": 1.27, "BM（高−低）": 1.14, "1月波动（低−高）": 0.81,
          "1月反转（低−高）": 1.47, "12月换手（低−高）": 0.33, "异常换手（低−高）": 1.14}

num = lambda x: pd.to_numeric(x.stack(), errors="coerce").unstack()
d = load_data(["a_market_value", "close_adj", "pe_ttm", "pb", "turn"], start="2005-01-01", end=E)
LSY_CFG = {"exclude_bottom_size": True}                        # LSY：剔市值最小 30%
mask = build_formation_mask(S, E, config=LSY_CFG)              # t 月末组建股票池
cols = mask.columns
ret_next = calc_monthly_returns(S, E, formation_config=LSY_CFG).shift(-1).reindex(index=mask.index, columns=cols)   # t → t+1

cap = _month_end_rows(num(d["a_market_value"])).reindex(index=mask.index, columns=cols).where(mask)

# 信号（t 月末可得）
pe = _month_end_rows(num(d["pe_ttm"])).reindex(index=mask.index, columns=cols)
pb = _month_end_rows(num(d["pb"])).reindex(index=mask.index, columns=cols)
ep, bm = 1 / pe.replace(0, np.nan), 1 / pb.replace(0, np.nan)
cadj = num(d["close_adj"]); r_d = cadj / cadj.shift(1) - 1
vol20 = _month_end_rows(r_d.rolling(20, min_periods=15).std()).reindex(index=mask.index, columns=cols)
rev20 = _month_end_rows(cadj / cadj.shift(20) - 1).reindex(index=mask.index, columns=cols)
# 日换手 = turn（框架权威来源）
tov = num(d["turn"])
t250 = tov.rolling(250, min_periods=120).mean(); t20 = tov.rolling(20, min_periods=15).mean()
turn12 = _month_end_rows(t250).reindex(index=mask.index, columns=cols)
abnt = _month_end_rows(t20 / t250).reindex(index=mask.index, columns=cols)


def vw(sel: pd.DataFrame) -> pd.Series:
    """组合 t+1 月市值加权收益（权重 = t 月末 A 股市值）。"""
    w = cap.where(sel & ret_next.notna())
    return (w * ret_next).sum(axis=1) / w.sum(axis=1)


def decile_ls(sig: pd.DataFrame, high_good: bool) -> pd.Series:
    s = sig.where(mask & cap.notna())
    q = s.rank(axis=1, pct=True)
    top, bot = vw(q > 0.9), vw(q <= 0.1)
    return (top - bot) if high_good else (bot - top)


# ── CH-3 ──
big = cap.ge(cap.median(axis=1), axis=0); small = cap.notna() & ~big
e = ep.where(mask & cap.notna()); qe = e.rank(axis=1, pct=True)
V, G = qe > 0.7, qe <= 0.3; M = (qe > 0.3) & (qe <= 0.7)
p = {f"{a}/{b}": vw(x & y) for a, x in (("S", small), ("B", big)) for b, y in (("V", V), ("M", M), ("G", G))}
SMB = (p["S/V"] + p["S/M"] + p["S/G"]) / 3 - (p["B/V"] + p["B/M"] + p["B/G"]) / 3
VMG = (p["S/V"] + p["B/V"]) / 2 - (p["S/G"] + p["B/G"]) / 2
dep = pd.read_csv("/Users/louis/MyProjects/Database/data/macro/deposit_rate_1y.csv", index_col=0, parse_dates=True).iloc[:, 0]
rf = (dep.reindex(dep.index.union(mask.index)).sort_index().ffill().reindex(mask.index) / 100 / 12)
MKT = vw(mask & cap.notna()) - rf
ch3 = pd.DataFrame({"MKT": MKT, "SMB": SMB, "VMG": VMG})

anom = pd.DataFrame({
    "size（小−大）": decile_ls(cap, False), "EP（高−低）": decile_ls(ep, True), "BM（高−低）": decile_ls(bm, True),
    "1月波动（低−高）": decile_ls(vol20, False), "1月反转（低−高）": decile_ls(rev20, False),
    "12月换手（低−高）": decile_ls(turn12, False), "异常换手（低−高）": decile_ls(abnt, False)})


def summ(x: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    y = x.loc[a:b].dropna(how="all")
    return pd.DataFrame({"均值%/月": y.mean() * 100, "标准差%": y.std() * 100,
                         "t": y.mean() / y.std() * np.sqrt(y.count()), "月数": y.count()})


pd.set_option("display.width", 220)
for a, b, lab in [("2006-01-31", "2016-11-30", "与 LSY 重叠期（组建 2006-01 ~ 2016-11，收益月 2006-02 ~ 2016-12）"),
                  ("2006-01-31", "2026-08-31", "全样本（收益月 2006-02 ~ 2026-09）")]:
    t3 = summ(ch3, a, b); t3["LSY 均值"] = pd.Series(LSY_T3)
    t6 = summ(anom, a, b); t6["LSY 均值"] = pd.Series(LSY_T6)
    print(f"\n==== {lab} ====\nCH-3（对照 LSY Table 3，2000–2016）：\n{t3.round(2).to_string()}")
    print(f"\n异象十分组市值加权多空（对照 LSY Table 6 Panel A）：\n{t6.round(2).to_string()}")
    print("CH-3 相关系数：\n", ch3.loc[a:b].corr().round(2).to_string())
ch3.to_csv("/Users/louis/MyProjects/Quant/output/A/stats/lsy_replication_ch3.csv")
anom.to_csv("/Users/louis/MyProjects/Quant/output/A/stats/lsy_replication_anomalies.csv")
