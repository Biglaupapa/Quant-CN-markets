"""
量化因子回测脚本（内存优化版）

使用方式：
    python run_backtest.py

说明：
  - 逐因子加载数据，避免同时驻留大文件（约 1.5 GB+）
  - 数据来源：存档 CSV（_archive/raw_data/，2014-2021）+ Database（OHLCV，2003-至今）
  - 回测区间：2014-01-01 ~ 2021-03-31（存档数据覆盖范围）
  - 可测因子：reversal_20, momentum_12_1, turnover_20, volatility_30,
             roll_spread, cs_spread, amihud_A, pb
  - 不可测因子（数据不足）：pe_ttm, dividend_yield（DB 仅 6 天快照），
                            net_profit_yoy（存档仅 52 只），ps/ap/ivol/ff3 系列

修复日志：
  v1.1 - 修复 inf% 问题：月初开盘价为零时做分母导致收益率无穷大，
          加入 replace(0, nan) 过滤
       - 修复路径：archive 改为相对路径，DB 使用实际路径
"""
import gc, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── 路径配置（修改 DB_PATH 指向 Database 实际位置）────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
ARCHIVE     = SCRIPT_DIR / "_archive" / "raw_data"
DB          = Path("/Users/louisliu/Mirror/MyProjects/Database/data/stock/A")
INDUSTRY_H5 = ARCHIVE / "FactorLoading_Industry_arch.h5"

START = "2014-01-01"
END   = "2021-03-31"
N     = 5        # 分组数
MIN_V = 10       # 滚动窗口内最少有效交易日

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def read_arch(fname, numeric=True, usecols=None):
    """
    读取存档 CSV（index=股票代码, columns=日期 YYYYMMDD）
    转置后返回：index=日期, columns=股票代码
    """
    path = ARCHIVE / fname
    df = pd.read_csv(path, index_col=0, low_memory=False, usecols=usecols)
    df = df.T
    df = df[df.index != "日期"]           # 去掉 "日期" 行（存档格式 artifact）
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index().loc[START:END]
    if numeric:
        df = df.apply(pd.to_numeric, errors="coerce")
    return df


def read_db(fname):
    """读取 Database CSV（index=日期, columns=股票代码）"""
    path = DB / fname
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().loc[START:END]
    return df.apply(pd.to_numeric, errors="coerce")


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def spearman(x, y):
    """纯 numpy/pandas 实现 Spearman 秩相关（不依赖 scipy）"""
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    n  = len(x)
    if n < 3:
        return np.nan
    num = np.cov(rx, ry)[0, 1]
    den = rx.std(ddof=1) * ry.std(ddof=1)
    return num / den if den > 0 else np.nan


def ols_resid(X, y):
    """OLS 残差（纯 numpy，不依赖 sklearn）"""
    try:
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ c
    except Exception:
        return np.full_like(y, np.nan, dtype=float)


def winsor(df, n=3):
    """横截面 3σ 截断去极值"""
    r = df.copy()
    for d in r.index:
        row = r.loc[d].dropna()
        if len(row) < 5:
            continue
        mu, sd = row.mean(), row.std()
        r.loc[d] = r.loc[d].clip(mu - n * sd, mu + n * sd)
    return r


def zscore(df):
    """横截面 Z-score 标准化"""
    mu  = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(std, axis=0)


def monthly_last(df):
    return df.resample("ME").last()


def apply_mask(df, mask):
    """将 mask=False 的位置置 NaN"""
    ci  = df.index.intersection(mask.index)
    cc  = df.columns.intersection(mask.columns)
    out = df.loc[ci, cc].copy()
    out[~mask.loc[ci, cc]] = np.nan
    return out


def neut_size(fm, log_mc):
    """市值中性化：横截面 OLS 对 log(市值) 做残差"""
    res = fm.copy() * np.nan
    ci  = fm.index.intersection(log_mc.index)
    cc  = fm.columns.intersection(log_mc.columns)
    for d in ci:
        y = fm.loc[d, cc]
        x = log_mc.loc[d, cc]
        ok = y.notna() & x.notna()
        if ok.sum() < MIN_V:
            continue
        X = np.column_stack([np.ones(ok.sum()), x[ok].values])
        res.loc[d, cc[ok]] = ols_resid(X, y[ok].values)
    return res


def neut_size_ind(fm, log_mc):
    """
    市值 + 行业双重中性化（行业哑变量来自 FactorLoading_Industry_arch.h5）
    若 H5 不可用则自动退化为纯市值中性化
    """
    if not INDUSTRY_H5.exists():
        print("    [警告] 行业 H5 文件不存在，降级为纯市值中性化")
        return neut_size(fm, log_mc)

    res = fm.copy() * np.nan
    ci  = fm.index.intersection(log_mc.index)
    cc  = fm.columns.intersection(log_mc.columns)

    try:
        store = pd.HDFStore(str(INDUSTRY_H5), mode="r")
        keys  = store.keys()
    except Exception as e:
        print(f"    [警告] 无法打开行业 H5（{e}），降级为纯市值中性化")
        return neut_size(fm, log_mc)

    for d in ci:
        ds  = d.strftime("%Y-%m-%d")
        y   = fm.loc[d, cc]
        sz  = log_mc.loc[d, cc]
        ok  = y.notna() & sz.notna()
        if ok.sum() < MIN_V:
            continue
        ys, szs = y[ok].values, sz[ok].values
        inds = []
        for k in keys:
            try:
                idf = store[k]
                if ds in idf.index:
                    inds.append(idf.loc[ds].reindex(cc[ok]).fillna(0).values)
            except Exception:
                continue
        parts = [np.ones(ok.sum()), szs]
        if inds:
            im = np.column_stack(inds)
            im = im[:, im.sum(0) > 0]
            if im.shape[1] > 0:
                parts.append(im)
        X = np.column_stack(parts)
        res.loc[d, cc[ok]] = ols_resid(X, ys)

    store.close()
    return res


def preprocess(df, neut=None, log_mc=None):
    """完整预处理管线：截断 → [中性化] → Z-score"""
    f = winsor(df)
    if   neut == "size":     f = neut_size(f, log_mc)
    elif neut == "size+ind": f = neut_size_ind(f, log_mc)
    return zscore(f)


# ─── 回测引擎 ─────────────────────────────────────────────────────────────────

def backtest(factor_m, monthly_ret, label):
    """
    分组回测：因子值按月末排名分为 N 组，计算各组下月收益率。

    G1 = 因子值最低组，G5 = 最高组，LS = G5 - G1 多空对冲
    """
    ci = factor_m.index.intersection(monthly_ret.index)
    cc = factor_m.columns.intersection(monthly_ret.columns)
    f  = factor_m.reindex(index=ci, columns=cc)
    r  = monthly_ret.reindex(index=ci, columns=cc)
    rf = r.shift(-1)   # 下期收益（forward return）

    grp = {f"G{i}": [] for i in range(1, N + 1)}
    grp["LS"] = []
    ics = []
    valid_dates = []

    for d in ci:
        sc  = f.loc[d].dropna()
        ret = rf.loc[d].reindex(sc.index).dropna()
        sc  = sc.reindex(ret.index).dropna()
        ret = ret.reindex(sc.index)
        if len(sc) < N * 5:
            continue
        try:
            lbl = pd.qcut(sc, q=N, labels=False, duplicates="drop")
        except Exception:
            continue
        valid_dates.append(d)
        for g in range(N):
            stk = lbl[lbl == g].index
            v   = ret.reindex(stk).dropna()
            grp[f"G{g+1}"].append(v.mean() if len(v) > 0 else np.nan)
        top = lbl[lbl == N - 1].index
        bot = lbl[lbl == 0].index
        grp["LS"].append(
            ret.reindex(top).dropna().mean() - ret.reindex(bot).dropna().mean()
        )
        ic = spearman(sc.values, ret.values)
        if not np.isnan(ic):
            ics.append(ic)

    ml  = min(len(v) for v in grp.values())
    vd  = valid_dates[:ml]
    gdf = pd.DataFrame({k: v[:ml] for k, v in grp.items()}, index=vd)
    ics = pd.Series(ics)

    def ar(s):
        s = s.dropna()
        return (1 + s).prod() ** (12 / len(s)) - 1 if len(s) > 0 else np.nan

    def av(s):
        return s.dropna().std() * np.sqrt(12)

    def sh(s):
        r, v = ar(s), av(s)
        return r / v if v and v > 0 else np.nan

    def md(s):
        c = (1 + s.fillna(0)).cumprod()
        return ((c - c.cummax()) / c.cummax()).min()

    def wr(s):
        s = s.dropna()
        return (s > 0).mean() if len(s) > 0 else np.nan

    rows = {}
    for col in gdf.columns:
        s = gdf[col]
        rows[col] = {
            "年化收益": f"{ar(s):.2%}",
            "年化波动": f"{av(s):.2%}",
            "夏普":     f"{sh(s):.2f}",
            "最大回撤": f"{md(s):.2%}",
            "胜率":     f"{wr(s):.1%}",
        }
    summary = pd.DataFrame(rows).T

    ic_m  = ics.mean()
    ic_s  = ics.std()
    ic_ir = ic_m / ic_s if ic_s > 0 else np.nan

    ls_row = summary.loc["LS"] if "LS" in summary.index else {}
    print(f"\n{'='*62}")
    print(f"  {label}  （有效月份：{ml}）")
    print(f"{'='*62}")
    print(summary.to_string())
    print(
        f"\n  ► RankIC={ic_m:.4f}  ICIR={ic_ir:.4f}  "
        f"IC>0占比={(ics > 0).mean():.1%}  "
        f"多空年化={ls_row.get('年化收益', 'N/A')}  "
        f"多空夏普={ls_row.get('夏普', 'N/A')}"
    )

    return {
        "ic_mean":   ic_m,
        "icir":      ic_ir,
        "ic_pos":    (ics > 0).mean(),
        "ls_ret":    ar(gdf["LS"]),
        "ls_sharpe": sh(gdf["LS"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 预加载：股票池掩码 + 月度收益 + 市值（各因子共用）
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  加载基础数据（掩码 + 月度收益 + 市值）")
print("=" * 62)

print("  loading is_st, trade_status, listing_days ...")
is_st    = read_arch("是否ST股.csv",     numeric=True)   # 0/1
trade_st = read_arch("交易状态.csv",     numeric=False)  # "交易"/"停牌" 字符串
list_d   = read_arch("上市交易日数.csv", numeric=True)   # 天数

is_st_n = is_st.fillna(0)
list_n  = list_d
is_trd  = (trade_st == "交易")
# 股票池过滤：非ST + 正常交易 + 上市满 60 交易日
mask = (~(is_st_n == 1)) & is_trd & (list_n >= 60)
mask = mask.fillna(False)
del is_st, trade_st, list_d, is_st_n, list_n, is_trd
gc.collect()
print(f"  掩码: {mask.shape}")

print("  loading close_adj, open_adj ...")
close_adj = read_arch("后复权收盘价.csv")
open_adj  = read_arch("后复权开盘价.csv")

# 月度收益率 = 月末收盘价 / 月初开盘价 - 1
# 修复 v1.1：open_m_first 可能含零（涨停开盘等特殊情况），
#            需先替换为 NaN，避免月度收益率出现 inf
open_m_first = apply_mask(open_adj, mask).resample("ME").first().replace(0, np.nan)
close_m_last = monthly_last(apply_mask(close_adj, mask))
prev_close_m = close_m_last.shift(1)

monthly_ret  = close_m_last / open_m_first - 1
# 过滤涨停开盘（月初开盘价 ≥ 上月收盘 × 1.099 视为一字涨停，收益不可实现）
monthly_ret[open_m_first >= prev_close_m * 1.099] = np.nan

print(f"  月度收益: {monthly_ret.shape}  "
      f"非空率={monthly_ret.notna().mean().mean():.1%}")

# 日度收盘（供因子计算，已过滤股票池）
close_clean = apply_mask(close_adj, mask)
del open_adj
gc.collect()

print("  loading close_raw, float_shares ...")
close_raw = read_arch("不复权收盘价.csv")
float_sh  = read_arch("流通股本.csv")
close_raw_m = monthly_last(apply_mask(close_raw, mask))
float_m     = monthly_last(apply_mask(float_sh, mask))
mktcap_m    = (close_raw_m * float_m).replace(0, np.nan)
log_mc      = np.log(mktcap_m)
del close_raw, float_sh
gc.collect()

print("\n  ✓ 基础数据就绪")
summary_all = {}


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 1：短期反转（Reversal 20）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n--- 1/8  短期反转 Reversal_20 ---")
valid = close_clean.rolling(20).count()
cf    = close_clean.copy()
cf[valid < MIN_V] = np.nan
ret20 = cf / cf.shift(20) - 1
f = preprocess(monthly_last(ret20))
summary_all["反转_20日"] = backtest(f, monthly_ret, "短期反转 Reversal_20")
del ret20, f
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 2：中期动量（Momentum 12-1）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n--- 2/8  中期动量 Momentum_12_1 ---")
close_m = monthly_last(close_clean)
mom = close_m.shift(1) / close_m.shift(12) - 1
f   = preprocess(mom)
summary_all["动量_12_1"] = backtest(f, monthly_ret, "中期动量 Momentum_12-1")
del mom, f
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 3：换手率（Turnover 20，市值中性化）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n--- 3/8  换手率 Turnover_20（市值中性化）---")
turn_db = read_db("turn.csv")
if turn_db is not None:
    turn_db = turn_db.reindex(columns=mask.columns)
    turn_m  = apply_mask(turn_db, mask)
    vt      = turn_m.rolling(20).count()
    turn_m[vt < MIN_V] = np.nan
    turn20 = monthly_last(turn_m.rolling(20).mean())
    f = preprocess(turn20, neut="size", log_mc=log_mc)
    summary_all["换手率_20日"] = backtest(f, monthly_ret, "换手率 Turnover_20（市值中性）")
    del turn_db, turn_m, turn20, f
    gc.collect()
else:
    print("  ✗ turn.csv 不可用")


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 4：短期波动率（Volatility 30）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n--- 4/8  短期波动率 Volatility_30 ---")
ret_d = close_clean / close_clean.shift(1) - 1
vol30 = monthly_last(ret_d.rolling(30).std())
f = preprocess(vol30)
summary_all["波动率_30日"] = backtest(f, monthly_ret, "短期波动率 Volatility_30")
del vol30, f
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 5：Roll 价差
# ═══════════════════════════════════════════════════════════════════════════════
# Roll(1984): S = 2 * sqrt(-Cov(r_t, r_{t-1}))，仅在 Cov < 0 时有意义，否则置 0
# 高 Roll 价差 = 流动性差。因子取负值后，G5 = 流动性最好。
print("\n\n--- 5/8  Roll 价差 ---")
roll_f, roll_d = [], []
for pe, grp in ret_d.resample("ME"):
    if len(grp) < MIN_V:
        roll_f.append(pd.Series(np.nan, index=grp.columns))
        roll_d.append(pe)
        continue
    row = {}
    for col in grp.columns:
        s = grp[col].dropna()
        if len(s) < MIN_V:
            row[col] = np.nan
            continue
        cov = np.cov(s.values[1:], s.values[:-1])[0, 1]
        row[col] = 2 * np.sqrt(-cov) if cov < 0 else 0.0
    roll_f.append(pd.Series(row))
    roll_d.append(pe)

roll_df = pd.DataFrame(roll_f, index=pd.DatetimeIndex(roll_d))
# 取负：低 Roll 价差（流动性好）→ 高因子值 → G5
f = preprocess(-roll_df)
summary_all["Roll价差"] = backtest(f, monthly_ret, "Roll 价差")
del roll_f, roll_df, f
gc.collect()
del ret_d
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 6：CS 价差（Corwin-Schultz，用 Database high/low）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n--- 6/8  CS 价差（Corwin-Schultz）---")
high = read_db("high.csv")
low  = read_db("low.csv")
if high is not None and low is not None:
    cc_cs = (close_clean.columns
             .intersection(high.columns)
             .intersection(low.columns))
    h = apply_mask(high.reindex(columns=cc_cs),
                   mask.reindex(columns=cc_cs)).replace(0, np.nan)
    l = apply_mask(low.reindex(columns=cc_cs),
                   mask.reindex(columns=cc_cs)).replace(0, np.nan)
    hl2   = (np.log(h / l)) ** 2
    hl2_1 = hl2.shift(-1)
    h2    = h.combine(h.shift(-1), np.fmax)
    l2    = l.combine(l.shift(-1), np.fmin)
    gam   = (np.log(h2 / l2)) ** 2
    beta  = hl2 + hl2_1
    k     = 3 - 2 * np.sqrt(2)
    alph  = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gam / k)
    spr   = (2 * (np.exp(alph) - 1) / (1 + np.exp(alph))).clip(lower=0)
    csf, csd = [], []
    for pe, grp in spr.resample("ME"):
        vc  = grp.count()
        row = grp.mean()
        row[vc < MIN_V] = np.nan
        csf.append(row)
        csd.append(pe)
    cs_df = (pd.DataFrame(csf, index=pd.DatetimeIndex(csd))
             .reindex(columns=monthly_ret.columns))
    # 取负：低 CS 价差（流动性好）→ 高因子值 → G5
    f = preprocess(-cs_df)
    summary_all["CS价差"] = backtest(f, monthly_ret, "CS 价差（Corwin-Schultz）")
    del high, low, h, l, hl2, gam, beta, alph, spr, csf, cs_df, f
    gc.collect()
else:
    print("  ✗ high/low 数据不可用")


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 7：Amihud 非流动性（版本A：3月滚动，成交额口径，滞后 1 月）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n--- 7/8  Amihud 非流动性（版本A）---")
amt = read_db("amt.csv")
if amt is not None:
    cc_a = close_clean.columns.intersection(amt.columns)
    c_a  = close_clean.reindex(columns=cc_a)
    a_a  = apply_mask(
        amt.reindex(columns=cc_a),
        mask.reindex(columns=cc_a)
    ).replace(0, np.nan)
    absr   = (c_a / c_a.shift(1) - 1).abs()
    dilliq = absr / a_a * 1e5    # 成交额单位：千元，×1e5 缩放
    buf, af, ad = [], [], []
    for pe, grp in dilliq.resample("ME"):
        buf.append(grp)
        buf = buf[-3:]           # 保留最近 3 个月
        if len(buf) < 3:
            continue
        window = pd.concat(buf)
        vc  = window.count()
        row = window.mean()
        row[vc < MIN_V] = np.nan
        af.append(row)
        ad.append(pe)
    amdf = (pd.DataFrame(af, index=pd.DatetimeIndex(ad))
            .shift(1)            # 滞后 1 月，避免前瞻偏差
            .reindex(columns=monthly_ret.columns))
    # 取负：低 Amihud（流动性好）→ 高因子值 → G5
    f = preprocess(-amdf)
    summary_all["Amihud_A"] = backtest(f, monthly_ret, "Amihud 非流动性（版本A）")
    del amt, c_a, a_a, absr, dilliq, af, amdf, f
    gc.collect()
else:
    print("  ✗ amt.csv 不可用")


# ═══════════════════════════════════════════════════════════════════════════════
# 因子 8：PB（市值+行业双重中性化）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n--- 8/8  市净率 PB（市值+行业双重中性化）---")
pb   = read_arch("PB.csv")
pb_m = monthly_last(apply_mask(pb, mask)).apply(pd.to_numeric, errors="coerce")
del pb
gc.collect()
f = preprocess(pb_m, neut="size+ind", log_mc=log_mc)
summary_all["PB_双重中性"] = backtest(f, monthly_ret, "市净率 PB（市值+行业中性化）")
del pb_m, f
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 综合排名
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 62)
print("  【因子综合排名】  回测区间：2014-01 ~ 2021-03")
print("═" * 62)

rank_rows = {}
for name, r in summary_all.items():
    rank_rows[name] = {
        "RankIC均值": f"{r['ic_mean']:+.4f}",
        "ICIR":       f"{r['icir']:+.4f}" if not np.isnan(r["icir"]) else "N/A",
        "IC>0占比":   f"{r['ic_pos']:.1%}",
        "多空年化":   f"{r['ls_ret']:+.2%}" if not np.isnan(r["ls_ret"]) else "N/A",
        "多空夏普":   f"{r['ls_sharpe']:+.2f}" if not np.isnan(r["ls_sharpe"]) else "N/A",
    }

rank_df = pd.DataFrame(rank_rows).T
rank_df["_icir_abs"] = (
    pd.to_numeric(rank_df["ICIR"].replace("N/A", np.nan), errors="coerce").abs()
)
rank_df = rank_df.sort_values("_icir_abs", ascending=False).drop(columns="_icir_abs")

print(rank_df.to_string())
print("═" * 62)
print("  注：所有因子均做 3σ 截断去极值 + 横截面 Z-score 标准化")
print("  换手率：市值中性化；PB：市值+行业（H5）双重中性化")
print("  不可测因子：pe_ttm / dividend_yield（DB 仅 6 天快照）")
print("              net_profit_yoy（存档仅 52 只，样本不足）")
print("              ps_gamma / ap_betas / ivol / ff3（需补充数据）")
