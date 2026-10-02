"""
Fama-French 五因子（FF5）构建模块 —— Fama & French (2015)，针对中国 A 股

与 FF3 共用股票池与全部机制（ff3_builder 不改，本模块只复用其函数）：
  股票池       SH + SZ（不含 BJ），可投资 = status==1 & st==0 & listed_days>=60
  Size 排序    6 月末流通市值，SH 股票中位数为 breakpoint
  组合加权     滞后一期流通市值加权；持有期 t 年 7 月 → t+1 年 6 月（年度再平衡）
  MKT / Rf     与 FF3 完全相同

新增两个排序变量（各自与 Size 做独立 2×3 排序，SH 股票 30/70 分位为 breakpoint）：
  OP  营业利润率 = (营业收入 − 营业成本 − 销售费用 − 管理费用 − 研发费用 − 财务费用) / 归母权益
      取 t−1 年年报。FF2015：(revenue − COGS − SG&A − interest) / BE；Compustat 的 SG&A 含 R&D，
      而 A 股 2018 年起研发费用从管理费用中单列，故一并扣除以保持口径一致。BE ≤ 0 剔除
  INV 资产增长 = t−1 年年报总资产 / t−2 年年报总资产 − 1
  时点：t 年 6 月 30 日前已公告的最新版本（财汇原始报表，按公告日，REPORTTYPE 1/3）

因子：
  HML = (SH+BH)/2 − (SL+BL)/2（与 FF3 相同的 B/M 排序）
  RMW = (S/R+B/R)/2 − (S/W+B/W)/2          R = 高 OP，W = 低 OP
  CMA = (S/C+B/C)/2 − (S/A+B/A)/2          C = 低 INV，A = 高 INV
  SMB = (SMB_BM + SMB_OP + SMB_INV) / 3   （FF2015）

输出：Database/data/factors/ff5_monthly.csv、ff5_daily.csv   列: MKT, SMB, HML, RMW, CMA, Rf
运行：python -m src.ff5_builder
"""
import logging
import sys
import time

import numpy as np
import pandas as pd

from src.ff3_builder import (DB_DATA_DIR, DB_FACTORS_DIR, _to_bool, load_data,
                             build_investable, calc_port_returns_period, calc_mkt)

sys.path.insert(0, "/Users/louis/MyProjects/Database/src")
from vendor_pit import load_statement            # noqa: E402  财汇报表读取 + 公告日兜底（同源）

log = logging.getLogger(__name__)
MAP_PATH = "/Users/louis/MyProjects/Database/data/mappings/security_map.csv"
OP_FIELDS = ["BIZINCO", "BIZCOST", "SALESEXPE", "MANAEXPE", "DEVEEXPE", "FINEXPE"]


def _fy_value(df: pd.DataFrame, field: str, fy: int, cutoff: str, c2s: dict) -> pd.Series:
    """FY 年报（ENDDATE = fy1231）在 cutoff 当日可见的最新版本 → Series(choice_code)。"""
    x = df[(df.ENDDATE == f"{fy}1231") & (df.PUBLISHDATE <= cutoff)].dropna(subset=[field])
    x = x.sort_values("PUBLISHDATE").drop_duplicates("COMPCODE", keep="last")
    s = x.set_index("COMPCODE")[field].astype(float)
    s.index = s.index.map(c2s)
    return s[s.index.notna()]


def characteristics(years) -> dict:
    """{year: DataFrame[OP, INV]}，year 为持有年（7 月 year 起）。"""
    smap = pd.read_csv(MAP_PATH, dtype=str).dropna(subset=["caihui_compcode"])
    c2s = dict(zip(smap.caihui_compcode, smap.choice_code))
    inc = load_statement("income_stmt_caihui.csv", OP_FIELDS)
    bal = load_statement("balance_sheet_caihui.csv", ["PARESHARRIGH", "TOTASSET"])
    out = {}
    for y in years:
        cut = f"{y}0630"
        f = {k: _fy_value(inc, k, y - 1, cut, c2s) for k in OP_FIELDS}
        be = _fy_value(bal, "PARESHARRIGH", y - 1, cut, c2s)
        ta1 = _fy_value(bal, "TOTASSET", y - 1, cut, c2s)
        ta2 = _fy_value(bal, "TOTASSET", y - 2, cut, c2s)
        idx = f["BIZINCO"].index
        cost = sum(f[k].reindex(idx).fillna(0) for k in OP_FIELDS[1:])  # 缺失的费用项按 0（未披露）
        op = (f["BIZINCO"] - cost) / be.reindex(idx).where(lambda b: b > 0)
        inv = ta1 / ta2.where(ta2 > 0) - 1
        out[y] = pd.DataFrame({"OP": op, "INV": inv})
    return out


def sort_2x3(size: pd.Series, char: pd.Series, valid: pd.Index, sh_codes: set,
             lo: str, hi: str) -> dict:
    """独立 2×3：size 用 SH 中位数，char 用 SH 30/70 分位。返回 {S?/B?: [codes]}。"""
    v = valid[size.reindex(valid).notna().to_numpy() & char.reindex(valid).notna().to_numpy()]
    sh = [c for c in v if c in sh_codes]
    if len(v) < 30 or len(sh) < 10:
        return {}
    sbp = size[sh].median()
    c30, c70 = char[sh].quantile(0.30), char[sh].quantile(0.70)
    sz = np.where(size[v] <= sbp, "S", "B")
    ch = np.where(char[v] <= c30, lo, np.where(char[v] >= c70, hi, "M"))
    lab = pd.Series([a + b for a, b in zip(sz, ch)], index=v)
    return {k: lab[lab == k].index.tolist() for k in [a + b for a in "SB" for b in (lo, "M", hi)]}


def build_ff5():
    t0 = time.time()
    all_cols = pd.read_csv(DB_DATA_DIR / "close_adj.csv", index_col=0, nrows=0).columns
    sh_codes = {c for c in all_cols if c.endswith(".SH")}
    sh_sz = sh_codes | {c for c in all_cols if c.endswith(".SZ")}

    d = load_data(sh_sz)
    close_adj, neg_mv, pb, bond = d["close_adj"], d["neg_mv"], d["pb"], d["bond_yld"]
    investable = build_investable(d["status"], d["st"], d["listed"])

    ret_d, lag_w_d = close_adj.pct_change(), neg_mv.shift(1)
    ret_m = close_adj.resample("ME").last().pct_change()
    lag_w_m = neg_mv.resample("ME").last().shift(1)
    rf_d = (bond / 100.0 / 252.0).reindex(ret_d.index).ffill().rename("Rf")
    rf_m = (bond.resample("ME").last() / 100.0 / 12.0).rename("Rf")
    neg_mv_me, pb_me = neg_mv.resample("ME").last(), pb.resample("ME").last()
    inv_me = _to_bool(_to_bool(investable.reindex(neg_mv.index)).resample("ME").last())

    years = range(neg_mv_me.index.year.min() + 1, neg_mv_me.index.year.max() + 1)
    chars = characteristics(years)

    parts_m, parts_d = [], []
    for y in years:
        june = neg_mv_me[(neg_mv_me.index.year == y) & (neg_mv_me.index.month == 6)].index
        dec = pb_me[(pb_me.index.year == y - 1) & (pb_me.index.month == 12)].index
        if len(june) == 0 or len(dec) == 0:
            continue
        size = neg_mv_me.loc[june[-1]]
        inv_ok = inv_me.asof(june[-1])
        valid = size.index[(inv_ok.reindex(size.index) == True).to_numpy() & size.notna().to_numpy()]
        bm = (1.0 / pb_me.loc[dec[-1]]).replace([np.inf, -np.inf], np.nan)
        bm = bm.where(bm > 0)
        c = chars[y]
        p_bm = sort_2x3(size, bm, valid, sh_codes, "L", "H")
        p_op = sort_2x3(size, c["OP"], valid, sh_codes, "W", "R")
        p_in = sort_2x3(size, c["INV"], valid, sh_codes, "C", "A")
        if not (p_bm and p_op and p_in):
            log.warning("  %d：某一排序有效股票不足，跳过", y)
            continue
        log.info("  %d：BM %d / OP %d / INV %d 只", y, sum(map(len, p_bm.values())),
                 sum(map(len, p_op.values())), sum(map(len, p_in.values())))
        ports = {**{"bm_" + k: v for k, v in p_bm.items()}, **{"op_" + k: v for k, v in p_op.items()},
                 **{"in_" + k: v for k, v in p_in.items()}}
        hs, he = pd.Timestamp(f"{y}-07-01"), pd.Timestamp(f"{y + 1}-07-01")
        mi = ret_m.index[(ret_m.index >= hs) & (ret_m.index < he)]
        di = ret_d.index[(ret_d.index >= hs) & (ret_d.index < he)]
        if len(mi):
            parts_m.append(calc_port_returns_period(ret_m, lag_w_m, ports, mi))
        if len(di):
            parts_d.append(calc_port_returns_period(ret_d, lag_w_d, ports, di))

    def factors(p: pd.DataFrame) -> pd.DataFrame:
        g = lambda pre, a: p[pre + a]
        smb = lambda pre, lo, hi: ((g(pre, "S" + lo) + g(pre, "SM") + g(pre, "S" + hi)) / 3
                                   - (g(pre, "B" + lo) + g(pre, "BM") + g(pre, "B" + hi)) / 3)
        return pd.DataFrame({
            "SMB": (smb("bm_", "L", "H") + smb("op_", "W", "R") + smb("in_", "C", "A")) / 3,
            "HML": (g("bm_", "SH") + g("bm_", "BH")) / 2 - (g("bm_", "SL") + g("bm_", "BL")) / 2,
            "RMW": (g("op_", "SR") + g("op_", "BR")) / 2 - (g("op_", "SW") + g("op_", "BW")) / 2,
            "CMA": (g("in_", "SC") + g("in_", "BC")) / 2 - (g("in_", "SA") + g("in_", "BA")) / 2,
        })

    fm = factors(pd.concat(parts_m).sort_index())
    fd = factors(pd.concat(parts_d).sort_index())
    mkt_m = calc_mkt(ret_m, lag_w_m, inv_me, rf_m)
    mkt_d = calc_mkt(ret_d, lag_w_d, _to_bool(investable.reindex(neg_mv.index)), rf_d)

    def save(mkt, f, rf, name):
        df = pd.concat([mkt, f, rf], axis=1).dropna(subset=["SMB"])
        df.index.name = "date"
        df.to_csv(DB_FACTORS_DIR / name)
        return df

    monthly = save(mkt_m, fm, rf_m, "ff5_monthly.csv")
    daily = save(mkt_d, fd, rf_d, "ff5_daily.csv")
    log.info("FF5 完成（%.0fs）：月度 %s → %s，%d 月", time.time() - t0,
             monthly.index[0].date(), monthly.index[-1].date(), len(monthly))
    log.info("月均 %%：%s", (monthly[["MKT", "SMB", "HML", "RMW", "CMA"]].mean() * 100).round(3).to_dict())
    return monthly, daily


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    build_ff5()
