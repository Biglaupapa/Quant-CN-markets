# -*- coding: utf-8 -*-
"""
CH-3 / CH-4 因子构建 —— Liu, Stambaugh, Yuan (2019, JFE) "Size and Value in China"
==================================================================================
输出（月度，index = 收益月末，单位为小数）：
    Database/data/factors/ch3_monthly.csv   MKT / SMB / VMG / RF
    Database/data/factors/ch4_monthly.csv   MKT / SMB / VMG / PMO / RF

方法（逐条对应 LSY 原文；与原文的差异见 docs/【方法】回测口径与文献对齐.md §十一、§十三）
--------------------------------------------------------------------------------------
- 股票池：组建日规则 a–g（settings.FORMATION_CONFIG）并**打开 g**（剔 A 股市值最小 30%）
- 市值：t 月末 a_market_value（财汇 TOTMKTCAP，A 股市值含限售股）。原文为「收盘 × 总股本，
  含非流通股」，未写明是否仅 A 股——按 Louis 2026-10-02 决定用 A 股市值，歧义已记录
- EP：1 / pe_mrq_deducted（财汇 PEMRQNPAAEI：最新报告期年化、扣非，按公告日更新）。
  负 EP 保留、参与排序（落入低端 = 成长组），与原文「categorize them as growth stocks」一致
- 异常换手：过去 20 日平均日换手 ÷ 过去 250 日平均日换手（turn = VOLUME / LIQSHARE；
  原文分母为总股本，比值形式下差异仅来自窗口内股本变动）
- 组合：市值中位数分 S / B；EP（或异常换手）按 30 / 40 / 30 分组；6 个市值加权组合，每月调仓
    SMB_EP = (S/V+S/M+S/G)/3 − (B/V+B/M+B/G)/3
    VMG    = (S/V+B/V)/2 − (S/G+B/G)/2
    PMO    = (S/低换手+B/低换手)/2 − (S/高换手+B/高换手)/2      （做多悲观 = 低异常换手）
    CH-3 SMB = SMB_EP；CH-4 SMB = (SMB_EP + SMB_TO)/2（原文：simple average，同 FF2015）
- MKT：股票池市值加权收益 − 一年期存款利率 / 12（RF 同列输出）
- 收益：t+1 月收盘到收盘、后复权、持有期不筛选（engine.calc_monthly_returns）

用法：cd ~/MyProjects/Quant && python -m src.ch_builder
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.data.universe import build_formation_mask, _month_end_rows
from src.backtest.engine import calc_monthly_returns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR  = Path("/Users/louis/MyProjects/Database/data/factors")
DEP_PATH = Path("/Users/louis/MyProjects/Database/data/macro/deposit_rate_1y.csv")
START, LOAD_START = "2006-01-01", "2005-01-01"     # 250 日窗口需往前一年
LSY_CFG = {"exclude_bottom_size": True}            # LSY：剔市值最小 30%


def _num(x: pd.DataFrame) -> pd.DataFrame:
    return x.apply(pd.to_numeric, errors="coerce")


def build(end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = build_formation_mask(START, end, config=LSY_CFG)          # t 月末组建
    cols = mask.columns
    ret = calc_monthly_returns(START, end, formation_config=LSY_CFG)  # 第 t 行 = t 月收益（组建于 t−1）
    ret_next = ret.shift(-1).reindex(index=mask.index, columns=cols)  # 第 t 行 = t+1 月收益

    d = load_data(["a_market_value", "pe_mrq_deducted", "turn"], start=LOAD_START, end=end)
    me = lambda x: _month_end_rows(_num(x)).reindex(index=mask.index, columns=cols)
    cap = me(d["a_market_value"]).where(mask)
    pe = me(d["pe_mrq_deducted"])
    ep = (1.0 / pe.where(pe != 0)).where(mask & cap.notna())
    tov = _num(d["turn"])
    abn = (tov.rolling(20, min_periods=15).mean() / tov.rolling(250, min_periods=120).mean())
    abn = _month_end_rows(abn).reindex(index=mask.index, columns=cols).where(mask & cap.notna())

    def vw(sel: pd.DataFrame) -> pd.Series:
        w = cap.where(sel & ret_next.notna())
        return (w * ret_next).sum(axis=1) / w.sum(axis=1)

    big = cap.ge(cap.median(axis=1), axis=0)
    small = cap.notna() & ~big

    def two_by_three(sig: pd.DataFrame):
        q = sig.rank(axis=1, pct=True)
        hi, lo = q > 0.7, q <= 0.3
        mid = (q > 0.3) & (q <= 0.7)
        p = {f"{a}/{b}": vw(x & y) for a, x in (("S", small), ("B", big))
             for b, y in (("H", hi), ("M", mid), ("L", lo))}
        smb = (p["S/H"] + p["S/M"] + p["S/L"]) / 3 - (p["B/H"] + p["B/M"] + p["B/L"]) / 3
        return p, smb

    p_ep, smb_ep = two_by_three(ep)
    vmg = (p_ep["S/H"] + p_ep["B/H"]) / 2 - (p_ep["S/L"] + p_ep["B/L"]) / 2
    p_to, smb_to = two_by_three(abn)
    pmo = (p_to["S/L"] + p_to["B/L"]) / 2 - (p_to["S/H"] + p_to["B/H"]) / 2

    dep = pd.read_csv(DEP_PATH, index_col=0, parse_dates=True).iloc[:, 0]
    # t+1 月的无风险利率：取 t 月末已知的一年期存款利率（年化 %）/ 12
    rf = dep.reindex(dep.index.union(mask.index)).sort_index().ffill().reindex(mask.index) / 100 / 12
    mkt = vw(mask & cap.notna()) - rf

    ch3 = pd.DataFrame({"MKT": mkt, "SMB": smb_ep, "VMG": vmg, "RF": rf})
    ch4 = pd.DataFrame({"MKT": mkt, "SMB": (smb_ep + smb_to) / 2, "VMG": vmg, "PMO": pmo, "RF": rf})
    for x in (ch3, ch4):
        x.index = x.index + pd.offsets.MonthEnd(1)        # 改为收益月
        x.index.name = "month_end"
    keep = ch3[["MKT", "SMB", "VMG"]].notna().all(axis=1)
    return ch3[keep], ch4[keep & ch4["PMO"].notna()]


def main() -> int:
    end = str((pd.Timestamp.today().to_period("M") - 1).to_timestamp("M").date())
    ch3, ch4 = build(end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ch3.to_csv(OUT_DIR / "ch3_monthly.csv", float_format="%.8f")
    ch4.to_csv(OUT_DIR / "ch4_monthly.csv", float_format="%.8f")
    for name, x in (("CH-3", ch3), ("CH-4", ch4)):
        log.info("%s %s → %s，%d 个月；月均 %%：%s", name, x.index.min().date(), x.index.max().date(),
                 len(x), (x.drop(columns="RF").mean() * 100).round(2).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
