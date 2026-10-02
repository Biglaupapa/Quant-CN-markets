# -*- coding: utf-8 -*-
"""
会计类因子 —— 数据源为财汇三张原始报表的 PIT 快照层
=====================================================
对标 JKP(Jensen-Kelly-Pedersen) 特征库命名，补上本框架此前完全缺失的
「会计块」。既有 62 个特征全部是量价与估值倍数衍生，唯一的财报类因子
`net_profit_yoy` 还依赖 2014~2021 的旧存档。

数据来自 `Database/data/vendor/caihui/pit/`，由 `Database/src/vendor_pit.py`
从三张原始报表按**公告日**展开而成。

为什么不用财汇现成的指标表
--------------------------
财汇有四张算好的指标表（`proindicdata` 313 列 / `proqindic` 144 列单季 /
`prottmindic` 146 列 TTM / `profinmainindex` 54 列），但**一张都不能用**：

- `prottmindic`：值被重述改写，公告日却停在首披日 → 前视且无字段可检测（17.6%）
- `proindicdata` / `proqindic`：首披/终披两日期塌陷。用首披日 → 5.8% 污染；
  用终披日 → **89% 的观测推迟整整一年**（几乎每份年报都会在次年年报里
  作为比较期例行重发，间隔中位数 365 天），数据直接废掉

只有三张原始报表没有这个权衡：`REPORTTYPE` 1（合并当期）与 3（合并重述）
各占一行、各挂自己的公告日，两版都在正确的日期上。

**这些现成表仍用于验算**——当期截面比对，PIT 缺陷在那里不咬人。
2026-09-02 逐因子对拍结果（截面 T=2025-05-31，约 5200~5500 只股票）：

| 我们的因子 | 财汇对照列 | Spearman | ≤1% 内 | 判读 |
|---|---|---|---|---|
| `ca_cl` 流动比率 | `CURRENTRT` | 0.9999 | **99.6%** | ✅ 逐位吻合 |
| `caliq_cl` 速动比率 | `QUICKRT` | 0.9999 | **99.6%** | ✅ 同时验证了稀疏科目补零 |
| `ocf_debt` | `OPNCFTOTLIAB` | 0.9960 | **98.8%** | ✅ 仅量纲差（财汇不乘 100）|
| `ocf_ni` | `OPNCFTONP` | 0.9770 | **97.3%** | ✅ 财汇用归母净利，已对齐 |
| `ocf_at` | `OPNCFTOTA` | 0.9961 | 96.7% | ✅ |
| `ni_be` ROE | `ROEDILUTED` | 0.9961 | 87.4% | ✅ |
| `sale_at`/`sale_inv`/`sale_rec` | `*TURNRT` | 0.93~0.97 | — | ⚠️ 财汇用**当期累计**非 TTM（倍数中位 1.012），**我们的 TTM 版更适合做因子**——财汇口径一年内阶梯跳 4 倍 |
| `niq_at` ROA | `ROA` | 0.9807 | — | ⚠️ 财汇分子含息、分母另有定义；**分子分母已各自独立验过**，故非 bug |

**方法论**：验算不追求"和财汇一模一样"，而是**定位差异的来源**。
差异若能被口径解释、且我们的口径更适合因子构建，就保留自己的。

三张报表三种口径
----------------
| 报表 | 口径 | 单季还原 |
|---|---|---|
| 利润表 / 现金流量表 | 累计（BEGINDATE 恒为 0101） | 差分 |
| 资产负债表 | **时点** | **禁止差分** |

搞错不会报错，只会静默出错值。本模块用 `_ttm()` / `_q()` / `_yoy()`
三个辅助函数把口径固化下来，各因子不再自己拼公式。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.settings import INDUSTRY_H5_PATH
from src.data.loader import load_data
from src.data.universe import build_investable_mask
from src.factors.base import preprocess

log = logging.getLogger(__name__)

PIT_DIR = Path("/Users/louis/MyProjects/Database/data/vendor/caihui/pit")


# ── PIT 面板读取与口径辅助 ────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _pit(field: str, variant: str = "cur") -> pd.DataFrame:
    """读一张 PIT 面板（月末 × 股票）。variant ∈ {cur, py_same, py_ann}。"""
    path = PIT_DIR / f"{field}__{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 PIT 面板 {path.name}——先在 Database 侧跑 "
            f"`python3 src/vendor_pit.py --fields {field}`"
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _ttm(field: str) -> pd.DataFrame:
    """累计口径字段的 TTM：本期累计 + 上年年报 − 上年同期累计。"""
    return _pit(field, "cur") + _pit(field, "py_ann") - _pit(field, "py_same")


def _avg(field: str) -> pd.DataFrame:
    """时点口径字段的期初期末均值（与 TTM 分子配比时用）。"""
    return (_pit(field, "cur") + _pit(field, "py_same")) / 2


def _yoy(field: str) -> pd.DataFrame:
    """同比增长率（JKP 的 `_gr1` 后缀）。分母取绝对值，避免基期为负时符号翻转。"""
    cur, base = _pit(field, "cur"), _pit(field, "py_same")
    return (cur - base) / base.abs()


# 资产负债表上「没有就不列这一行」的科目：读出来是 NaN，语义其实是 0。
# 不补 0 的话，无长期借款的公司会整个因子缺失——实测 `debtlt_gr1a` 非空率
# 只有 24.4%、`noa_at` 26.7%，就是这么丢的。
SPARSE_ZERO = {"LONGBORR", "SHORTTERMBORR", "INVE", "ACCORECE", "NOTESRECE",
               "ACQUASSETCASH"}


def _z(field: str, variant: str = "cur") -> pd.DataFrame:
    """稀疏科目读取：缺失即 0。"""
    df = _pit(field, variant)
    return df.fillna(0) if field in SPARSE_ZERO else df


def _chg_at(field: str) -> pd.DataFrame:
    """
    同比变化额 ÷ 平均总资产（JKP 的 `_gr1a` 后缀，`a` = scaled by assets）。

    ⚠️ 这**不是**百分比增长率。JKP 对存货、应收、借款、税项这类科目一律用
    「变化额除以总资产」而非增长率——因为基期可能是 0 或接近 0，
    增长率会爆表。首版写成了 `_yoy`，是定义错误，已改正。
    """
    d = _z(field, "cur") - _z(field, "py_same")
    return d / _avg("TOTASSET")


def _mktcap(start, end) -> pd.DataFrame:
    """月末总市值（Choice），用于分母是市值的那几个因子。"""
    mv = load_data(["market_value"], start=start, end=end)["market_value"]
    return mv.resample("ME").last()


def _align(df: pd.DataFrame, mask_m: pd.DataFrame) -> pd.DataFrame:
    """对齐到投资域月末掩码，域外置 NaN。"""
    idx = df.index.intersection(mask_m.index)
    col = df.columns.intersection(mask_m.columns)
    out = df.reindex(index=idx, columns=col)
    out[~mask_m.reindex(index=idx, columns=col)] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


# ── 因子定义表 ────────────────────────────────────────────────────────────────
#
# 每条：JKP 名 → (公式, 是否做市值+行业中性化, 中文说明)
# 公式是无参 lambda，返回「月末 × 股票」的 DataFrame。
#
# EBIT 用 `TOTPROFIT + FINEXPE` 近似（利润总额 + 财务费用）。财汇利润表没有
# 单列的利息支出，财务费用是最接近的代理，含汇兑损益因而略有噪声。

SPECS: dict[str, tuple] = {
    # ── 盈利能力 ──
    "ni_be":        (lambda: _ttm("PARENETP") / _pit("PARESHARRIGH"),   True,  "ROE（TTM归母净利/归母权益）"),
    "niq_at":       (lambda: _ttm("NETPROFIT") / _pit("TOTASSET"),      True,  "ROA（TTM净利/总资产）"),
    "ope_be":       (lambda: _ttm("PERPROFIT") / _pit("PARESHARRIGH"),  True,  "营业利润/归母权益"),
    "ebit_sale":    (lambda: (_ttm("TOTPROFIT") + _ttm("FINEXPE")) / _ttm("BIZINCO"), True, "EBIT利润率"),
    "gp_at":        (lambda: (_ttm("BIZINCO") - _ttm("BIZCOST")) / _pit("TOTASSET"), True, "毛利/总资产（GPA）"),
    "netmargin":    (lambda: _ttm("PARENETP") / _ttm("BIZINCO"),        True,  "净利率"),
    "pi_nix":       (lambda: _ttm("TOTPROFIT") / _ttm("NETPROFIT"),     True,  "税前利润/净利润"),

    # ── 应计与盈利质量（与量价正交性最高的一组）──
    # ⚠️ `.abs()` 只用在 `_yoy` 的**基期**上（基期为负时增长率符号会翻转）。
    # **比值的分母绝不能取 abs**——分母符号有经济含义。首版给 `ocf_ni` /
    # `taccruals_ni` 的分母加了 abs，27.7% 的公司归母净利为负、符号被翻，
    # 逐截面 Spearman 从 0.985 掉到 0.712。
    "oaccruals_at": (lambda: (_ttm("PARENETP") - _ttm("MANANETR")) / _pit("TOTASSET"), True, "经营应计/总资产（Sloan 1996）"),
    "taccruals_ni": (lambda: (_ttm("NETPROFIT") - _ttm("MANANETR")) / _ttm("NETPROFIT"),       True, "总应计/净利润"),
    "ocf_at":       (lambda: _ttm("MANANETR") / _pit("TOTASSET"),       True,  "经营现金流/总资产"),
    "ocf_debt":     (lambda: _ttm("MANANETR") / _pit("TOTLIAB"),        True,  "经营现金流/总负债"),
    "ocf_ni":       (lambda: _ttm("MANANETR") / _ttm("PARENETP"),       True, "经营现金流/归母净利（现金含量）"),

    # ── 增长与投资 ──
    "at_gr1":       (lambda: _yoy("TOTASSET"),                          True,  "总资产增长率"),
    "be_gr1a":      (lambda: _chg_at("PARESHARRIGH"),                      True,  "归母权益增长率"),
    "sale_gr1":     (lambda: _yoy("BIZINCO"),                           True,  "营收增长率"),
    "ni_gr1":       (lambda: _yoy("PARENETP"),                          True,  "归母净利增长率"),
    "inv_gr1a":     (lambda: _chg_at("INVE"),                              True,  "存货增长率"),
    "rec_gr1a":     (lambda: _chg_at("ACCORECE"),                          True,  "应收账款增长率"),
    "cash_gr1a":    (lambda: _chg_at("CURFDS"),                            True,  "货币资金增长率"),
    "debt_gr1":     (lambda: _yoy("TOTLIAB"),                           True,  "总负债增长率"),
    "debtlt_gr1a":  (lambda: _chg_at("LONGBORR"),                          True,  "长期借款增长率"),
    "tax_gr1a":     (lambda: _chg_at("INCOTAXEXPE"),                       True,  "所得税费用增长率"),
    "capx_gr1a":    (lambda: _chg_at("ACQUASSETCASH"),                     True,  "资本支出增长率"),
    "capx_at":      (lambda: _ttm("ACQUASSETCASH") / _pit("TOTASSET"),  True,  "资本支出强度"),

    # ── 杠杆与偿债 ──
    "lev":          (lambda: _pit("TOTLIAB") / _pit("TOTASSET"),        True,  "资产负债率"),
    "ca_cl":        (lambda: _pit("TOTCURRASSET") / _pit("TOTALCURRLIAB"), True, "流动比率"),
    "caliq_cl":     (lambda: (_pit("TOTCURRASSET") - _pit("INVE")) / _pit("TOTALCURRLIAB"), True, "速动比率"),
    "cash_at":      (lambda: _pit("CURFDS") / _pit("TOTASSET"),         True,  "现金占总资产"),
    "noa_at":       (lambda: (_pit("TOTASSET") - _pit("CURFDS")
                              - (_pit("TOTLIAB") - _z("SHORTTERMBORR") - _z("LONGBORR")))
                             / _pit("TOTASSET"),                        True,  "净经营资产/总资产"),

    # ── 运营效率 ──
    "sale_at":      (lambda: _ttm("BIZINCO") / _pit("TOTASSET"),        True,  "总资产周转率"),
    "sale_inv":     (lambda: _ttm("BIZCOST") / _avg("INVE"),            True,  "存货周转率"),
    "sale_rec":     (lambda: _ttm("BIZINCO") / _avg("ACCORECE"),        True,  "应收账款周转率"),

    # ── 研发 ──
    #
    # ⚠️ `DEVEEXPE`（研发费用）在利润表里单列是 2018 年财报准则修订后的事，
    # 之前多数公司计入管理费用。所以这两个因子在 2018 年前覆盖极低，
    # 属数据真实性质而非缺陷；ML 面板的缺失率过滤会自行处理。
    "rd_sale":      (lambda: _ttm("DEVEEXPE") / _ttm("BIZINCO"),        True,  "研发费用/营收"),
    "rd_at":        (lambda: _ttm("DEVEEXPE") / _pit("TOTASSET"),       True,  "研发费用/总资产"),

    # ── 盈余意外（事件驱动，量价因子在结构上捕捉不到）──
    #
    # ★ 这三个是本批最有价值的：现有 62 个特征全是量价与估值衍生，
    #   PEAD（盈余公告后漂移）这一类信号完全没有代表。
    "niq_su":       (lambda: _to_monthly(_sue("PARENETP")),             False, "标准化未预期盈余 SUE"),
    "saleq_su":     (lambda: _to_monthly(_sue("BIZINCO")),              False, "标准化未预期营收"),
    "ni_inc8q":     (lambda: _to_monthly(_consecutive_up("PARENETP")),  False, "连续同比增长季度数"),

    # ── 复合指标 ──
    "f_score":      (lambda: _f_score(),                                False, "Piotroski F-Score（九项二元信号）"),
    "qmj_prof":     (lambda: _qmj_prof(),                               False, "QMJ 盈利质量（六项 z 分均值）"),
    "capex_abn":    (lambda: _capex_abn(),                              True,  "异常资本支出（相对三年均值）"),

    # ── 销售与配套项目的背离（盈余管理/舞弊的经典代理）──
    "dsale_dinv":   (lambda: _yoy("BIZINCO") - _chg_at("INVE"),            True,  "营收增长 − 存货增长"),
    "dsale_drec":   (lambda: _yoy("BIZINCO") - _chg_at("ACCORECE"),        True,  "营收增长 − 应收增长"),
    "dsale_dsga":   (lambda: _yoy("BIZINCO") - _yoy("SALESEXPE"),       True,  "营收增长 − 销售费用增长"),
}

# 分母是市值的因子单列——它们要跨源取 Choice 的 market_value，签名不同
SPECS_MV: dict[str, tuple] = {
    "cfp":          (lambda mv: _ttm("MANANETR") / mv,                  "经营现金流/市值"),
    "fcf_me":       (lambda mv: (_ttm("MANANETR") - _ttm("ACQUASSETCASH")) / mv, "自由现金流/市值"),
    "debt_me":      (lambda mv: _pit("TOTLIAB") / mv,                   "总负债/市值"),
    "netdebt_me":   (lambda mv: (_pit("TOTLIAB") - _pit("CURFDS")) / mv, "净负债/市值"),
    "rd_me":        (lambda mv: _ttm("DEVEEXPE") / mv,                  "研发费用/市值"),
}


def _make(name: str):
    """把 SPECS 里的一条编译成标准 calc_ 函数（签名与其余因子模块一致）。"""
    def calc(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        formula, neutral, _ = SPECS[name]
        mask_m = build_investable_mask(start=start, end=end, freq="D") \
                     .resample("ME").last().fillna(False)
        raw = _align(formula(), mask_m)
        if not neutral:
            return preprocess(raw)
        mv = load_data(["neg_market_value"], start=start, end=end)["neg_market_value"]
        logmv = np.log(mv.resample("ME").last().replace(0, np.nan))
        return preprocess(raw, neutralize="size+industry",
                          log_mktcap=logmv.reindex_like(raw),
                          industry_h5_path=INDUSTRY_H5_PATH)
    calc.__name__ = f"calc_{name}"
    calc.__doc__ = f"{SPECS[name][2]}　[财汇 PIT]"
    return calc


def _make_mv(name: str):
    def calc(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        formula, _ = SPECS_MV[name]
        mask_m = build_investable_mask(start=start, end=end, freq="D") \
                     .resample("ME").last().fillna(False)
        mv = _mktcap(start, end)
        raw = _align(formula(mv.reindex_like(_pit("TOTASSET"))), mask_m)
        logmv = np.log(mv.replace(0, np.nan))
        return preprocess(raw, neutralize="size+industry",
                          log_mktcap=logmv.reindex_like(raw),
                          industry_h5_path=INDUSTRY_H5_PATH)
    calc.__name__ = f"calc_{name}"
    calc.__doc__ = f"{SPECS_MV[name][1]}　[财汇 PIT × Choice 市值]"
    return calc


ACCOUNTING_FACTORS = {n: _make(n) for n in SPECS}
ACCOUNTING_FACTORS.update({n: _make_mv(n) for n in SPECS_MV})

globals().update({f"calc_{n}": f for n, f in ACCOUNTING_FACTORS.items()})

__all__ = ["ACCOUNTING_FACTORS", "SPECS", "SPECS_MV", "PIT_DIR"]


# ── 报告期空间：单季序列与盈余意外 ────────────────────────────────────────────
#
# SUE 这类因子需要「过去 8 个季度的单季序列」，而月末 PIT 面板只给 cur/py_same/py_ann
# 三个切片。做法是换个空间算：
#
#   1. 在**报告期空间**（COMPCODE × ENDDATE）上取首次披露值，还原单季
#   2. 在报告期空间上算 SUE
#   3. 用 `_meta__enddate` 把结果映回月末
#
# 第 3 步保证了 PIT 正确性——`_meta__enddate` 本身就是「T 时刻可见的最新报告期」。
# 第 1 步用**首次披露值**（REPORTTYPE=1）而非最新重述版，这对 SUE 恰恰更对：
# 盈余意外衡量的是市场当时收到的意外，就该用当时公布的数。

RAW_DIR = Path("/Users/louis/MyProjects/Database/data/vendor/caihui")
_MAP = pd.read_csv(Path("/Users/louis/MyProjects/Database/data/mappings/security_map.csv"),
                   dtype=str).dropna(subset=["caihui_compcode"])
_C2S = dict(zip(_MAP.caihui_compcode, _MAP.choice_code))


@lru_cache(maxsize=32)
def _period_panel(field: str, fname: str = "income_stmt_caihui.csv") -> pd.DataFrame:
    """报告期 × 股票的**首次披露**累计值（REPORTTYPE=1）。"""
    df = pd.read_csv(RAW_DIR / fname,
                     usecols=["COMPCODE", "ENDDATE", "REPORTTYPE", "PUBLISHDATE", field],
                     dtype={"COMPCODE": str, "ENDDATE": str,
                            "REPORTTYPE": str, "PUBLISHDATE": str})
    df = df[df.REPORTTYPE == "1"].sort_values("PUBLISHDATE")
    df = df.drop_duplicates(subset=["COMPCODE", "ENDDATE"], keep="first")
    w = df.pivot(index="ENDDATE", columns="COMPCODE", values=field).rename(columns=_C2S)
    w = w.loc[:, [c for c in w.columns if "." in str(c)]]
    return w.sort_index()


def _single_quarter(field: str, fname: str = "income_stmt_caihui.csv") -> pd.DataFrame:
    """
    累计口径 → 单季。Q1 即累计值；Q2~Q4 = 本期累计 − 上期累计。

    ⚠️ Q4 单季常为负（年末计提减值），实测约 45% 的公司如此。
    **这是真实现象，不要"修正"它。**
    """
    cum = _period_panel(field, fname)
    q = cum.diff()                                   # 相邻报告期作差
    is_q1 = pd.Index(cum.index).str.endswith("0331")
    q.loc[is_q1] = cum.loc[is_q1]                    # Q1 的累计即单季
    # 报告期不连续时（漏报一期）差分无意义，置 NaN
    yr = pd.Index(cum.index).str[:4].astype(int)
    gap = pd.Series(cum.index, index=cum.index).shift(1).notna() & ~pd.Series(is_q1, index=cum.index)
    prev_yr = pd.Series(yr, index=cum.index).shift(1)
    q[~(pd.Series(is_q1, index=cum.index) | (prev_yr == yr))] = np.nan
    return q


def _to_monthly(period_df: pd.DataFrame) -> pd.DataFrame:
    """报告期空间 → 月末空间，按 `_meta__enddate`（T 时刻可见的最新报告期）映射。"""
    meta = pd.read_csv(PIT_DIR / "_meta__enddate.csv", index_col=0, parse_dates=True)
    meta = meta.astype(str).replace(r"\.0$", "", regex=True)
    cols = period_df.columns.intersection(meta.columns)
    out = pd.DataFrame(index=meta.index, columns=cols, dtype=float)
    lookup = {c: period_df[c] for c in cols}
    for c in cols:
        s = lookup[c]
        out[c] = meta[c].map(s)
    return out


def _sue(field: str, fname: str = "income_stmt_caihui.csv", window: int = 8) -> pd.DataFrame:
    """
    标准化未预期盈余（Foster 时序模型，Bernard-Thomas 1989）：

        SUE = (Q_t − Q_{t−4} − drift) / σ(Q − Q_{−4} 过去 window 期)

    用时序模型而非分析师一致预期：覆盖面更全（不依赖有无覆盖），
    且不引入分析师数据自身的时点问题。
    """
    q = _single_quarter(field, fname)
    d = q - q.shift(4)                                # 同比变化（去季节性）
    drift = d.rolling(window, min_periods=4).mean().shift(1)
    sigma = d.rolling(window, min_periods=4).std().shift(1)
    return ((d - drift) / sigma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _consecutive_up(field: str, max_q: int = 8) -> pd.DataFrame:
    """连续同比增长的季度数（JKP `ni_inc8q`），上限 max_q。"""
    q = _single_quarter(field)
    up = (q > q.shift(4)).astype(float).where(q.notna() & q.shift(4).notna())
    run = up.copy() * 0
    acc = pd.Series(0.0, index=up.columns)
    for i in range(len(up)):
        row = up.iloc[i]
        acc = np.where(row == 1, np.minimum(acc + 1, max_q), np.where(row == 0, 0, acc))
        run.iloc[i] = acc
    return run.where(up.notna())


# ── 复合指标 ──────────────────────────────────────────────────────────────────

def _xs_z(df: pd.DataFrame) -> pd.DataFrame:
    """逐截面 z 分。合成多个量纲不同的指标时必须先做，否则大方差项主导。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def _f_score() -> pd.DataFrame:
    """
    Piotroski(2000) F-Score：九项二元信号求和，0~9。

    原文九项与本实现的对应：
      盈利性 4 项：ROA>0、CFO>0、ΔROA>0、CFO>ROA（应计质量）
      杠杆/流动性 3 项：Δ长期负债率<0、Δ流动比率>0、未增发股本
      运营效率 2 项：Δ毛利率>0、Δ总资产周转>0

    「未增发股本」用总股本（市值÷收盘价）判断——Choice 没有单列的总股本字段。
    """
    ta, ta_p = _pit("TOTASSET"), _pit("TOTASSET", "py_same")
    roa   = _ttm("PARENETP") / ta
    roa_p = _pit("PARENETP", "py_same").div(ta_p)          # 上年同期累计/上年资产，近似
    cfo   = _ttm("MANANETR") / ta
    ltd   = _z("LONGBORR") / ta
    ltd_p = _z("LONGBORR", "py_same") / ta_p
    cr    = _pit("TOTCURRASSET") / _pit("TOTALCURRLIAB")
    cr_p  = _pit("TOTCURRASSET", "py_same") / _pit("TOTALCURRLIAB", "py_same")
    gm    = (_ttm("BIZINCO") - _ttm("BIZCOST")) / _ttm("BIZINCO")
    gm_p  = ((_pit("BIZINCO", "py_same") - _pit("BIZCOST", "py_same"))
             / _pit("BIZINCO", "py_same"))
    at    = _ttm("BIZINCO") / ta
    at_p  = _pit("BIZINCO", "py_same") / ta_p

    # 总股本直接读 Choice TOTALSHARE（2026-10-02 起；原为 MV ÷ close 绕回，见登记表）
    sh = load_data(["total_shares"])["total_shares"].resample("ME").last().reindex_like(ta)
    no_issue = (sh <= sh.shift(12) * 1.02)                 # 一年内股本增幅 ≤2% 视作未增发

    sig = [roa > 0, cfo > 0, roa > roa_p, cfo > roa,
           ltd < ltd_p, cr > cr_p, no_issue,
           gm > gm_p, at > at_p]
    valid = roa.notna() & cfo.notna() & ta_p.notna()
    return sum(x.astype(float) for x in sig).where(valid)


def _qmj_prof() -> pd.DataFrame:
    """
    Asness-Frazzini-Pedersen QMJ 的盈利性分项：
    六个指标各自逐截面 z 分后取均值。
    """
    ta = _pit("TOTASSET")
    parts = {
        "gpoa": (_ttm("BIZINCO") - _ttm("BIZCOST")) / ta,
        "roe":  _ttm("PARENETP") / _pit("PARESHARRIGH"),
        "roa":  _ttm("PARENETP") / ta,
        "cfoa": _ttm("MANANETR") / ta,
        "gmar": (_ttm("BIZINCO") - _ttm("BIZCOST")) / _ttm("BIZINCO"),
        "acc": -(_ttm("PARENETP") - _ttm("MANANETR")) / ta,   # 应计越低越好，取负
    }
    zs = [_xs_z(v.replace([np.inf, -np.inf], np.nan)) for v in parts.values()]
    return sum(zs) / len(zs)


def _capex_abn() -> pd.DataFrame:
    """
    异常资本支出：本期 capex/营收 相对过去三年均值的偏离。
    Titman-Wei-Xie(2004)：过度投资预示低回报。
    """
    ci = _ttm("ACQUASSETCASH") / _ttm("BIZINCO")
    base = ci.shift(12).rolling(24, min_periods=12).mean()   # 月频面板，12 期 ≈ 1 年
    return ci / base.replace(0, np.nan) - 1
