"""
Fama-French 三因子（FF3）构建模块

改编自 Database/src/ff3.py，路径指向 Database/data/
在 Quant 项目中运行，输出到 Database/data/factors/

方法论：Fama & French (1993)，针对中国 A 股
  股票池       SH + SZ，不含北交所 BJ
  Size 排序    6月末流通市值（neg_market_value），以 SH 股票中位数为 breakpoint
  B/M  排序    t-1年12月末 1/PB，以 SH 股票 30/70 分位为 breakpoint
  组合加权     滞后一期流通市值加权
  持有期       t 年 7月 → t+1 年 6月（年度再平衡）
  MKT          全 SH+SZ 可投资股票 VW 超额收益（独立计算，不限于6组合）
  无风险利率   bond_yield_1y（日度 /252，月度月末值 /12）

输出：
  /Users/louis/MyProjects/Database/data/factors/ff3_monthly.csv   列: date, MKT, SMB, HML, Rf
  /Users/louis/MyProjects/Database/data/factors/ff3_daily.csv     列: date, MKT, SMB, HML, Rf

运行：
  python -m src.ff3_builder            全量重建
  python -m src.ff3_builder --dry-run  查看当前输出文件状态
"""

import argparse
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ── 数据路径（指向 Database）──────────────────────────────────────────────────
DB_DATA_DIR      = Path("/Users/louis/MyProjects/Database/data/stock/A")
DB_MACRO_DIR     = Path("/Users/louis/MyProjects/Database/data/macro")
DB_FACTORS_DIR   = Path("/Users/louis/MyProjects/Database/data/factors")
STOCKS_LIST_PATH = DB_DATA_DIR / "stocks_list.csv"

# 创建输出目录
DB_FACTORS_DIR.mkdir(parents=True, exist_ok=True)

# ── 日志 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# 0. 工具函数
# =============================================================================

def _to_bool(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 转换为 bool dtype，NaN → False。
    """
    return pd.DataFrame(df.to_numpy() == True, index=df.index, columns=df.columns)


# =============================================================================
# 1. 数据加载
# =============================================================================

def _read_wide(fname: str, keep_cols: set = None) -> pd.DataFrame:
    """读取 A/ 目录下的宽格式 CSV，可选只保留指定股票列。"""
    df = pd.read_csv(DB_DATA_DIR / fname, index_col=0, parse_dates=True)
    if keep_cols is not None:
        cols = [c for c in df.columns if c in keep_cols]
        df = df[cols]
    return df


def load_data(sh_sz_codes: set) -> dict:
    log.info("加载原始数据...")
    t0 = time.time()

    data = {
        "close_adj": _read_wide("close_adj.csv",       sh_sz_codes),
        "neg_mv":    _read_wide("neg_market_value.csv", sh_sz_codes),
        "pb":        _read_wide("pb.csv",               sh_sz_codes),
        "status":    _read_wide("status.csv",           sh_sz_codes),
        "st":        _read_wide("st.csv",               sh_sz_codes),
        "listed":    _read_wide("listed_days.csv",      sh_sz_codes),
        "bond_yld":  pd.read_csv(
                         DB_MACRO_DIR / "bond_yield_1y.csv",
                         index_col=0, parse_dates=True
                     ).squeeze(),
    }

    log.info(
        f"加载完成 ({time.time()-t0:.1f}s)  "
        f"close_adj={data['close_adj'].shape}  "
        f"neg_mv={data['neg_mv'].shape}  "
        f"pb={data['pb'].shape}"
    )
    return data


# =============================================================================
# 2. 可投资掩码
# =============================================================================

def build_investable(status: pd.DataFrame,
                     st: pd.DataFrame,
                     listed: pd.DataFrame) -> pd.DataFrame:
    """
    可投资掩码：status==1，st==0，listed_days>=60。
    对齐三个来源的 index 和 columns，NaN 保守处理（视为不可投）。
    """
    idx  = status.index.intersection(st.index).intersection(listed.index)
    cols = status.columns.intersection(st.columns).intersection(listed.columns)

    s = status.reindex(index=idx, columns=cols).fillna(0)
    t = st.reindex(index=idx, columns=cols).fillna(1)   # NaN → 视为 ST
    l = listed.reindex(index=idx, columns=cols).fillna(0)

    return (s == 1) & (t == 0) & (l >= 60)


# =============================================================================
# 3. 年度组合排序（单年）
# =============================================================================

def assign_year(year: int,
                neg_mv_me: pd.DataFrame,
                pb_me: pd.DataFrame,
                investable_me: pd.DataFrame,
                sh_codes: set) -> dict:
    """
    对第 year 个持有年（7月year → 6月year+1）进行 2×3 排序。

    Size  breakpoint: SH 股票 6月末流通市值中位数
    B/M   breakpoints: SH 股票 12月末(year-1) B/M 的 30/70 分位

    Returns:
        {'SL': [codes], 'SM': [codes], 'SH': [codes],
         'BL': [codes], 'BM': [codes], 'BH': [codes]}
        若数据不足则返回空 dict。
    """
    # ── 6月末流通市值 ─────────────────────────────────────────
    june_dates = neg_mv_me[
        (neg_mv_me.index.year == year) & (neg_mv_me.index.month == 6)
    ].index
    if len(june_dates) == 0:
        return {}
    june_mv  = neg_mv_me.loc[june_dates[-1]]
    june_inv = investable_me.asof(june_dates[-1]) \
               if june_dates[-1] <= investable_me.index[-1] \
               else pd.Series(False, index=june_mv.index)

    # ── 12月末(year-1) B/M = 1/PB ─────────────────────────────
    dec_dates = pb_me[
        (pb_me.index.year == year - 1) & (pb_me.index.month == 12)
    ].index
    if len(dec_dates) == 0:
        return {}
    dec_pb  = pb_me.loc[dec_dates[-1]]
    bm_vals = (1.0 / dec_pb).replace([np.inf, -np.inf], np.nan)
    bm_vals[bm_vals <= 0] = np.nan          # 负 B/M 无意义

    # ── 有效股票：size + bm 均有值 + 可投资 ────────────────────
    common = (june_mv.index
              .intersection(bm_vals.index)
              .intersection(june_inv.index))
    valid_mask = (
        june_mv.reindex(common).notna() &
        bm_vals.reindex(common).notna() &
        pd.Series(june_inv.reindex(common).to_numpy() == True, index=common)
    )
    valid = valid_mask[valid_mask].index

    if len(valid) < 30:
        log.warning(f"  {year}: 有效股票不足（{len(valid)}只），跳过本年排序")
        return {}

    sh_valid = [c for c in valid if c in sh_codes]
    if len(sh_valid) < 10:
        log.warning(f"  {year}: SH breakpoint 股票不足（{len(sh_valid)}只），跳过")
        return {}

    # ── Breakpoints ────────────────────────────────────────────
    size_bp = june_mv[sh_valid].median()
    bm_30   = bm_vals[sh_valid].quantile(0.30)
    bm_70   = bm_vals[sh_valid].quantile(0.70)

    # ── 向量化标签赋值 ──────────────────────────────────────────
    mv_v  = june_mv.reindex(valid)
    bm_v  = bm_vals.reindex(valid)

    size_lbl = pd.Series(
        np.where(mv_v <= size_bp, 'S', 'B'), index=valid
    )
    bm_lbl = pd.Series(
        np.where(bm_v <= bm_30, 'L',
                 np.where(bm_v >= bm_70, 'H', 'M')),
        index=valid
    )

    ports = {}
    for sz in ['S', 'B']:
        for bm in ['L', 'M', 'H']:
            mask = (size_lbl == sz) & (bm_lbl == bm)
            ports[sz + bm] = mask[mask].index.tolist()

    counts = {k: len(v) for k, v in ports.items()}
    log.info(
        f"  {year}: {len(valid)} 只股票排序  "
        f"SL={counts['SL']} SM={counts['SM']} SH={counts['SH']} "
        f"BL={counts['BL']} BM={counts['BM']} BH={counts['BH']}  "
        f"size_bp={size_bp/1e8:.0f}亿  bm_bp=[{bm_30:.3f},{bm_70:.3f}]"
    )
    return ports


# =============================================================================
# 4. 组合收益率（单年，向量化）
# =============================================================================

def calc_port_returns_period(returns: pd.DataFrame,
                             lag_w: pd.DataFrame,
                             ports: dict,
                             date_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    计算持有期内 6 个组合的流通市值加权收益率。

    Args:
        returns:   全量收益率矩阵（日或月）
        lag_w:     全量滞后流通市值矩阵
        ports:     {port_name: [stock_codes]}
        date_idx:  持有期日期索引

    Returns:
        DataFrame，shape=(len(date_idx), 6)，列为 SL/SM/SH/BL/BM/BH
    """
    result = {}
    for port_name, stocks in ports.items():
        stks = [s for s in stocks if s in returns.columns]
        if not stks:
            result[port_name] = pd.Series(np.nan, index=date_idx)
            continue

        r = returns.loc[date_idx, stks]
        w = lag_w.loc[date_idx, stks].where(r.notna(), other=np.nan)

        w_sum  = w.sum(axis=1).replace(0, np.nan)
        rw_sum = (r * w).sum(axis=1)
        result[port_name] = rw_sum / w_sum

    return pd.DataFrame(result, index=date_idx)


# =============================================================================
# 5. SMB / HML / MKT
# =============================================================================

def calc_smb_hml(port_rets: pd.DataFrame) -> pd.DataFrame:
    """
    SMB = (SL + SM + SH) / 3 - (BL + BM + BH) / 3
    HML = (SH + BH) / 2    - (SL + BL) / 2
    """
    smb = ((port_rets['SL'] + port_rets['SM'] + port_rets['SH']) / 3
           - (port_rets['BL'] + port_rets['BM'] + port_rets['BH']) / 3)
    hml = ((port_rets['SH'] + port_rets['BH']) / 2
           - (port_rets['SL'] + port_rets['BL']) / 2)
    return pd.DataFrame({'SMB': smb, 'HML': hml})


def calc_mkt(returns: pd.DataFrame,
             lag_w: pd.DataFrame,
             investable: pd.DataFrame,
             rf: pd.Series) -> pd.Series:
    """
    全 SH+SZ 可投资股票流通市值加权收益率 - Rf（独立计算，不限于6组合股票）。
    """
    idx  = returns.index
    cols = returns.columns

    inv = _to_bool(investable.reindex(index=idx, columns=cols))
    r   = returns.where(inv)
    w   = lag_w.reindex(index=idx, columns=cols).where(
              inv & returns.notna(), other=np.nan
          )

    w_sum  = w.sum(axis=1).replace(0, np.nan)
    vw_ret = (r * w).sum(axis=1) / w_sum

    rf_aligned = rf.reindex(idx).ffill()
    return (vw_ret - rf_aligned).rename('MKT')


# =============================================================================
# 6. 主流程：build_ff3
# =============================================================================

def build_ff3():
    DB_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── 股票池 ────────────────────────────────────────────────
    stocks_df   = pd.read_csv(STOCKS_LIST_PATH)
    sh_codes    = set(stocks_df[stocks_df['exchange'] == 'SH']['code'])
    sz_codes    = set(stocks_df[stocks_df['exchange'] == 'SZ']['code'])
    sh_sz_codes = sh_codes | sz_codes
    log.info(f"股票池：SH {len(sh_codes)} + SZ {len(sz_codes)} = {len(sh_sz_codes)} 只")

    # ── 加载数据 ──────────────────────────────────────────────
    d = load_data(sh_sz_codes)

    close_adj = d["close_adj"]
    neg_mv    = d["neg_mv"]
    pb        = d["pb"]
    bond_yld  = d["bond_yld"]

    # ── 可投资掩码 ────────────────────────────────────────────
    log.info("构建可投资掩码...")
    investable = build_investable(d["status"], d["st"], d["listed"])

    # ── 收益率与权重（日度）──────────────────────────────────
    ret_d   = close_adj.pct_change()
    lag_w_d = neg_mv.shift(1)

    # ── 收益率与权重（月度）──────────────────────────────────
    ret_m    = close_adj.resample('ME').last().pct_change()
    neg_mv_m = neg_mv.resample('ME').last()
    lag_w_m  = neg_mv_m.shift(1)

    # ── 无风险利率 ────────────────────────────────────────────
    rf_d = (bond_yld / 100.0 / 252.0).reindex(ret_d.index).ffill().rename('Rf')
    rf_m = (bond_yld.resample('ME').last() / 100.0 / 12.0).rename('Rf')

    # ── 月末数据（用于年度排序）──────────────────────────────
    neg_mv_me    = neg_mv.resample('ME').last()
    pb_me        = pb.resample('ME').last()
    investable_me = _to_bool(
                        _to_bool(investable.reindex(neg_mv.index))
                        .resample('ME').last()
                    )

    # ── 年度循环：排序 + 计算组合收益率 ──────────────────────
    log.info("执行年度组合排序与组合收益率计算...")

    all_ports_m: list[pd.DataFrame] = []
    all_ports_d: list[pd.DataFrame] = []

    years = range(neg_mv_me.index.year.min() + 1, neg_mv_me.index.year.max() + 1)

    for year in years:
        ports = assign_year(year, neg_mv_me, pb_me, investable_me, sh_codes)
        if not ports:
            continue

        hold_start = pd.Timestamp(f'{year}-07-01')
        hold_end   = pd.Timestamp(f'{year + 1}-07-01')   # exclusive

        # 月度
        m_idx = ret_m.index[(ret_m.index >= hold_start) & (ret_m.index < hold_end)]
        if len(m_idx) > 0:
            all_ports_m.append(
                calc_port_returns_period(ret_m, lag_w_m, ports, m_idx)
            )

        # 日度
        d_idx = ret_d.index[(ret_d.index >= hold_start) & (ret_d.index < hold_end)]
        if len(d_idx) > 0:
            all_ports_d.append(
                calc_port_returns_period(ret_d, lag_w_d, ports, d_idx)
            )

    # ── 合并各年收益率 ────────────────────────────────────────
    port_rets_m = pd.concat(all_ports_m).sort_index()
    port_rets_d = pd.concat(all_ports_d).sort_index()

    # ── MKT（全市场，独立计算）──────────────────────────────
    log.info("计算 MKT 因子（全市场 VW）...")
    inv_m = _to_bool(
                _to_bool(investable.reindex(neg_mv.index))
                .resample('ME').last()
            )
    mkt_m = calc_mkt(ret_m, lag_w_m, inv_m, rf_m)

    inv_d = _to_bool(investable.reindex(neg_mv.index))
    mkt_d = calc_mkt(ret_d, lag_w_d, inv_d, rf_d)

    # ── SMB / HML ─────────────────────────────────────────────
    log.info("合成 SMB / HML 因子...")
    ff3_m = calc_smb_hml(port_rets_m)
    ff3_d = calc_smb_hml(port_rets_d)

    # ── 输出 ──────────────────────────────────────────────────
    def save(mkt, factors, rf, path):
        df = pd.concat([mkt, factors, rf], axis=1).dropna(how='all')
        df.index.name = 'date'
        df.to_csv(path)
        return df

    log.info("保存输出文件...")
    monthly = save(mkt_m, ff3_m, rf_m, DB_FACTORS_DIR / 'ff3_monthly.csv')
    daily   = save(mkt_d, ff3_d, rf_d, DB_FACTORS_DIR / 'ff3_daily.csv')

    # ── 汇总统计 ──────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info(f"FF3 构建完成（{elapsed:.1f}s）")
    log.info(f"月度：shape={monthly.shape}  "
             f"{monthly.index[0].date()} → {monthly.index[-1].date()}")
    log.info(f"日度：shape={daily.shape}  "
             f"{daily.index[0].date()} → {daily.index[-1].date()}")
    log.info("")
    log.info("月度因子统计（均值 / 标准差，%）：")
    desc = monthly[['MKT', 'SMB', 'HML']].mul(100).describe().loc[['mean', 'std']]
    for idx_name, row in desc.iterrows():
        log.info(f"  {idx_name:5s}: " + "  ".join(f"{k}={v:+.3f}" for k, v in row.items()))
    log.info("=" * 60)

    return monthly, daily


# =============================================================================
# 7. 状态查看 & CLI
# =============================================================================

def print_status():
    log.info("=" * 55)
    log.info("FF3 因子输出文件状态")
    log.info("=" * 55)
    for label, fname in [("月度", "ff3_monthly.csv"), ("日度", "ff3_daily.csv")]:
        path = DB_FACTORS_DIR / fname
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            log.info(
                f"  {label}  shape={df.shape}  "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
        else:
            log.info(f"  {label}  ❌ 尚未构建  ({path})")
    log.info("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Fama-French 三因子全量构建")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅查看当前输出文件状态，不执行计算"
    )
    args = parser.parse_args()

    if args.dry_run:
        print_status()
    else:
        build_ff3()


if __name__ == "__main__":
    main()
