# =============================================================================
# subperiod_analysis.py
# 子区间因子表现对比分析（后处理脚本，独立运行，不改动回测框架）
#
# 目的：
#   对比因子在「全样本」与「近期（2024+）」两个时间窗内的表现差异，
#   观察随量化 / AI 普及，因子收益是否发生衰减或反转。
#
# 数据来源（复用 main.py 已生成的月度序列，无需重跑回测）：
#   output/<market>/stats/<factor>_group_returns.csv   # 列含 G1..Gn, LS
#   output/<market>/stats/<factor>_ic_series.csv       # 列 IC
#
# 复用 src.backtest.metrics 的指标函数，口径与主报告完全一致。
#
# 运行方式（在 Quant/ 根目录）：
#   python -m src.subperiod_analysis
#
# 产出（output/<market>/subperiod/）：
#   factor_subperiod_comparison.csv   # 各因子 全样本 vs 2024+ 对比表
#   ls_annual_full_vs_2024.png        # 全样本 vs 2024+ LS 年化收益对比柱状图
#   multi_factor_rolling.png          # 多因子合成 36 月滚动年化收益 / Sharpe 曲线
# =============================================================================

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # 非交互后端，与 report.py 一致
import matplotlib.pyplot as plt

from src.config.settings import FACTOR_OUTPUT_DIR
from src.backtest.metrics import (
    annualized_return, annualized_volatility, sharpe_ratio,
    max_drawdown, win_rate,
)

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
MARKETS = ["A", "HK"]                          # 分析的市场
WINDOWS = [("全样本", None), ("2024+", "2024-01-01")]   # (标签, 起始日期; None=不设下限)
FREQ = 12                                      # 月度
ROLL_WINDOW = 36                               # 滚动年化窗口（月）
MIN_MONTHS = 6                                 # 窗口内少于此月数则指标记 NaN（样本太少）
ROLL_TARGET = "multi_factor_equal"             # 滚动曲线针对的因子（多因子等权合成）

# 因子方向（与 main.py 的 FACTOR_DIRECTIONS 一致）：
#   LS 收益类指标按方向对齐（负向因子 ×-1），使口径与主报告一致（如 turnover_20 显示 +年化）。
#   IC / ICIR 保持原始带符号，不对齐（与报告一致，如 turnover_20 IC 显示为负）。
#   合成因子及未列出者默认 +1（已对齐）。
FACTOR_DIRECTIONS = {
    "reversal_20": -1, "momentum_12_1": -1, "turnover_20": -1, "turnover_20_neutral": -1,
    "amihud": +1, "amihud_neutral": +1, "amihud_zero_adj": +1, "amihud_zero_adj_neutral": +1,
    "cs_spread": -1, "roll_spread": -1, "overnight_ret": +1, "volatility_30": -1,
    "pb": -1, "bm": +1, "pe_ttm": -1, "pe1": -1, "dividend_yield": +1,
    "size": -1, "size2": -1, "net_profit_yoy": +1,
    "ps_ttm": -1, "ev_ebitda": -1, "est_pe_ftm": -1, "est_peg": +1, "ev2_neutral": -1,
}


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def _read_series(path: Path, col: str) -> pd.Series:
    """读单列月度序列，index 转为日期。文件或列缺失返回空 Series。"""
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].dropna()


def _slice(series: pd.Series, start) -> pd.Series:
    """按起始日期切片（含）。start=None 表示全样本。"""
    if start is None:
        return series
    return series[series.index >= pd.Timestamp(start)]


def _ls_metrics(ls: pd.Series) -> dict:
    """LS 多空组合在给定窗口的绩效指标。"""
    if len(ls) < MIN_MONTHS:
        return {k: np.nan for k in
                ["LS年化", "LS夏普", "LS最大回撤", "月度胜率", "月数"]}
    return {
        "LS年化":     annualized_return(ls, FREQ),
        "LS夏普":     sharpe_ratio(ls, FREQ),
        "LS最大回撤": max_drawdown(ls),
        "月度胜率":   win_rate(ls),
        "月数":       int(len(ls)),
    }


def _ic_metrics(ic: pd.Series) -> dict:
    """IC 序列在给定窗口的均值与 ICIR。"""
    if len(ic) < MIN_MONTHS:
        return {"IC均值": np.nan, "ICIR": np.nan}
    std = ic.std()
    return {
        "IC均值": ic.mean(),
        "ICIR":   ic.mean() / std if std > 0 else np.nan,
    }


# -----------------------------------------------------------------------------
# 单市场分析
# -----------------------------------------------------------------------------
def analyze_market(market: str) -> pd.DataFrame | None:
    stats_dir = Path(FACTOR_OUTPUT_DIR) / market / "stats"
    out_dir   = Path(FACTOR_OUTPUT_DIR) / market / "subperiod"
    if not stats_dir.exists():
        print(f"[{market}] 未找到 {stats_dir}，跳过")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    # 发现所有含 group_returns 的因子
    factors = sorted(
        Path(p).name.replace("_group_returns.csv", "")
        for p in glob.glob(str(stats_dir / "*_group_returns.csv"))
    )
    if not factors:
        print(f"[{market}] stats 目录无 group_returns 文件，跳过")
        return None

    rows = []
    for f in factors:
        ls_full = _read_series(stats_dir / f"{f}_group_returns.csv", "LS")
        ic_full = _read_series(stats_dir / f"{f}_ic_series.csv", "IC")
        if ls_full.empty:
            continue
        # LS 按因子方向对齐（负向因子 ×-1），口径与主报告一致；IC 保持原始符号
        ls_full = ls_full * FACTOR_DIRECTIONS.get(f, 1)
        row = {"因子": f}
        for label, start in WINDOWS:
            lm = _ls_metrics(_slice(ls_full, start))
            im = _ic_metrics(_slice(ic_full, start))
            for k, v in {**lm, **im}.items():
                row[f"{label}_{k}"] = v
        # 衰减：近期 − 全样本（百分点）
        if pd.notna(row.get("2024+_LS年化")) and pd.notna(row.get("全样本_LS年化")):
            row["ΔLS年化(pp)"] = (row["2024+_LS年化"] - row["全样本_LS年化"]) * 100
        else:
            row["ΔLS年化(pp)"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("因子")
    # 按全样本 LS 夏普降序，便于阅读
    if "全样本_LS夏普" in df.columns:
        df = df.sort_values("全样本_LS夏普", ascending=False)

    # 保存对比表
    csv_path = out_dir / "factor_subperiod_comparison.csv"
    df.to_csv(csv_path, encoding="utf-8-sig", float_format="%.4f")
    print(f"[{market}] 对比表已保存：{csv_path}  （{len(df)} 个因子）")

    # 图 1：全样本 vs 2024+ LS 年化对比（水平分组柱状）
    _plot_bar(df, market, out_dir)
    # 图 2：多因子合成滚动年化曲线
    _plot_rolling(stats_dir, market, out_dir)
    return df


# -----------------------------------------------------------------------------
# 绘图（图内标签用英文，规避中文字体问题，与 report.py 一致）
# -----------------------------------------------------------------------------
def _plot_bar(df: pd.DataFrame, market: str, out_dir: Path) -> None:
    if "全样本_LS年化" not in df.columns or "2024+_LS年化" not in df.columns:
        return
    d = df.sort_values("全样本_LS年化")           # 从低到高，水平柱更好看
    factors = list(d.index)
    y = np.arange(len(factors))
    h = 0.4
    full = d["全样本_LS年化"].values * 100
    recent = d["2024+_LS年化"].values * 100

    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(factors))))
    ax.barh(y + h/2, full,   height=h, label="Full sample (2007-2026)", color="#1e3a5f")
    ax.barh(y - h/2, recent, height=h, label="2024+",                   color="#e0a030")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(factors, fontsize=9)
    ax.set_xlabel("LS Annualized Return (%)", fontsize=10)
    ax.set_title(f"[{market}] Factor LS Annualized Return: Full Sample vs 2024+",
                 fontsize=13, pad=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    p = out_dir / "ls_annual_full_vs_2024.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{market}] 柱状图已保存：{p}")


def _plot_rolling(stats_dir: Path, market: str, out_dir: Path) -> None:
    ls = _read_series(stats_dir / f"{ROLL_TARGET}_group_returns.csv", "LS")
    if ls.empty or len(ls) < ROLL_WINDOW + 3:
        print(f"[{market}] {ROLL_TARGET} 序列不足 {ROLL_WINDOW}+ 月，跳过滚动曲线")
        return
    roll_ret = ls.rolling(ROLL_WINDOW).apply(
        lambda x: annualized_return(pd.Series(x), FREQ), raw=False) * 100
    roll_shp = ls.rolling(ROLL_WINDOW).apply(
        lambda x: sharpe_ratio(pd.Series(x), FREQ), raw=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(roll_ret.index, roll_ret.values, color="#1e3a5f", lw=1.6)
    ax1.axhline(annualized_return(ls, FREQ) * 100, color="#e0a030", ls="--", lw=1,
                label="Full-sample annualized")
    ax1.set_ylabel("Rolling Ann. Return (%)", fontsize=10)
    ax1.set_title(f"[{market}] {ROLL_TARGET}: {ROLL_WINDOW}M Rolling Performance",
                  fontsize=13, pad=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(ls=":", alpha=0.5)
    ax1.axvspan(pd.Timestamp("2024-01-01"), ls.index.max(), color="#e0a030", alpha=0.08)

    ax2.plot(roll_shp.index, roll_shp.values, color="#2e7d5b", lw=1.6)
    ax2.axhline(sharpe_ratio(ls, FREQ), color="#e0a030", ls="--", lw=1)
    ax2.set_ylabel("Rolling Sharpe", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(ls=":", alpha=0.5)
    ax2.axvspan(pd.Timestamp("2024-01-01"), ls.index.max(), color="#e0a030", alpha=0.08)

    fig.tight_layout()
    p = out_dir / "multi_factor_rolling.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{market}] 滚动曲线已保存：{p}")


# -----------------------------------------------------------------------------
# 主入口
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("子区间因子表现对比：全样本  vs  2024+")
    print("=" * 70)
    for market in MARKETS:
        print(f"\n----- 市场：{market} -----")
        df = analyze_market(market)
        if df is not None:
            cols = [c for c in ["全样本_LS年化", "2024+_LS年化", "ΔLS年化(pp)",
                                "全样本_ICIR", "2024+_ICIR"] if c in df.columns]
            with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
                print(df[cols].to_string())
    print("\n完成。详见各市场 output/<market>/subperiod/ 目录。")


if __name__ == "__main__":
    main()
