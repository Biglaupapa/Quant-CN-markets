# =============================================================================
# backtest/report.py
# 绩效报告生成
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from src.backtest.metrics import group_summary, ic_summary

import matplotlib
matplotlib.use("Agg")          # 非交互后端，不弹窗，适合脚本/服务器运行
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =============================================================================
# 因子显示名称映射
#
# 图表标题使用 display name（首字母大写、去除窗口数字），
# 文件名和缓存键继续使用原始 factor_name，两者互不影响。
#
# 负向因子（direction=-1）中：
#   turnover → "Inverse Turnover"（常见学术命名，低换手=好）
#   其余负向因子保留语义名，不加 Inverse 前缀（约定俗成无此叫法）
# =============================================================================

FACTOR_DISPLAY_NAMES = {
    # ── 微观结构 ──────────────────────────────────────────────────────────────
    "reversal_20":               "Reversal",
    "momentum_12_1":             "Momentum",
    "turnover_20":               "Inverse Turnover",          # 负向→取反显示
    "turnover_20_neutral":       "Inverse Turnover (Neutral)",# 负向→取反显示
    "amihud":                    "Amihud",
    "amihud_neutral":            "Amihud (Neutral)",
    "amihud_zero_adj":           "Amihud (Zero Adj)",
    "amihud_zero_adj_neutral":   "Amihud (Zero Adj, Neutral)",
    "cs_spread":                 "CS Spread",
    "roll_spread":               "Roll Spread",
    "overnight_ret":             "Overnight Return",
    "volatility_30":             "Volatility",
    # ── 基本面 ────────────────────────────────────────────────────────────────
    "pb":                        "Price-to-Book",
    "bm":                        "Book-to-Market",
    "pe_ttm":                    "PE (TTM)",
    "pe1":                       "PE (Dynamic)",
    "dividend_yield":            "Dividend Yield",
    "size":                      "Size",
    "size2":                     "Size (Float)",
    "net_profit_yoy":            "Net Profit YoY",
    # ── 多因子合成 ────────────────────────────────────────────────────────────
    "multi_factor_equal":        "Multi-Factor (Equal Weight)",
    "multi_factor_icir":         "Multi-Factor (ICIR Weight)",
}


def _get_display_name(factor_name: str) -> str:
    """
    将因子内部名称转换为图表显示名称。

    优先查 FACTOR_DISPLAY_NAMES；未找到则通用回退：
    下划线换空格，每词首字母大写（title case）。
    """
    if factor_name in FACTOR_DISPLAY_NAMES:
        return FACTOR_DISPLAY_NAMES[factor_name]
    return factor_name.replace("_", " ").title()


# =============================================================================
# 报告打印
# =============================================================================

def print_factor_report(
    factor_name: str,
    group_ret: pd.DataFrame,
    ic_series: pd.Series,
    freq: int = 12,
) -> None:
    """
    打印因子回测报告（控制台输出）。

    Parameters
    ----------
    factor_name : str
        因子名称
    group_ret : pd.DataFrame
        分组月度收益（来自 engine.group_return()）
    ic_series : pd.Series
        月度 IC 序列（来自 metrics.calc_ic()）
    freq : int
        数据频率（月度=12，季度=4）
    """
    print("=" * 60)
    print(f"  因子回测报告：{factor_name}")
    print("=" * 60)

    print("\n【分组绩效统计】")
    summary = group_summary(group_ret, freq=freq)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n【IC 统计】")
    ic_stats = ic_summary(ic_series)
    for k, v in ic_stats.items():
        print(f"  {k}: {v:.4f}")

    print("=" * 60)


# =============================================================================
# CSV 报告保存
# =============================================================================

def save_report(
    factor_name: str,
    group_ret: pd.DataFrame,
    ic_series: pd.Series,
    output_dir: Path,
    freq: int = 12,
) -> None:
    """
    将回测结果保存为 CSV 文件。

    Parameters
    ----------
    output_dir : Path
        输出目录（通常为 FACTOR_OUTPUT_DIR）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    group_ret.to_csv(output_dir / f"{factor_name}_group_returns.csv")
    ic_series.to_csv(output_dir / f"{factor_name}_ic_series.csv", header=["IC"])

    summary = group_summary(group_ret, freq=freq)
    summary.to_csv(output_dir / f"{factor_name}_summary.csv")

    print(f"[report] 报告已保存至 {output_dir}/")


# =============================================================================
# NAV 曲线绘制
# =============================================================================

def plot_nav_curve(
    factor_name: str,
    group_ret: pd.DataFrame,
    output_dir: Path,
    freq: int = 12,
    dual_panel: bool = False,
    direction: int = 1,
) -> None:
    """
    绘制分组累计净值（NAV）曲线并保存为 PNG。

    NAV 定义
    --------
    NAV(T) = ∏(1 + r_t)，从 1.0 起始。
    图表纵轴是 NAV，不是收益率（收益率 = NAV - 1）。
    log scale（对数坐标）使纵轴等距 = 等比例涨跌，适合长期复利展示。

    direction 参数（负向因子 LS 翻转）
    -----------------------------------
    direction=+1（默认）：LS 按原始方向显示（G5 − G1）。
        适用于正向因子（IC > 0），LS 自然向上。
    direction=−1：LS 取反后显示（G1 − G5）。
        适用于负向因子（IC < 0）：原始 LS = G5 − G1 < 0，向下；
        取反后 LS > 0，向上，与"低因子值=好股票"的语义一致。
        G1~Gn 分组线不变（G1 仍是最低因子值的组）。
        会打印 [report] 翻转提示，明确记录哪些因子做了处理。

    dual_panel 参数
    ---------------
    False（默认，单因子）：单图，log scale，G1~Gn + LS 全部画在一起。
    True（多因子合成）  ：双子图，上图 G1~Gn log scale，下图 LS 线性。

    Parameters
    ----------
    factor_name : str
        因子内部名称（用于文件名），显示名称由 _get_display_name() 转换
    group_ret : pd.DataFrame
        分组月度收益，index=月末日期，columns=[G1,...,Gn,LS]
    output_dir : Path
        图片输出目录
    freq : int
        数据年化频率（月度=12，季度=4），用于计算 Sharpe
    dual_panel : bool
        单图 / 双子图模式
    direction : int
        +1 = 正向因子（LS 原始显示）；-1 = 负向因子（LS 取反显示）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    display_name = _get_display_name(factor_name)

    # ── 负向因子：LS 取反，使曲线向上，与经济含义一致 ─────────────────
    plot_ret = group_ret.copy()
    if direction == -1 and "LS" in plot_ret.columns:
        plot_ret["LS"] = -plot_ret["LS"]
        print(f"[report] {factor_name}（{display_name}）: "
              f"负向因子 direction=-1，LS 已翻转显示（G1−G5，方向向上）")

    # ── 累计 NAV（从 1.0 起始）
    # nav(T) = ∏(1 + r_t)：每期月度收益率复乘，结果即净值
    nav = (1 + plot_ret.fillna(0)).cumprod()

    group_cols = [c for c in plot_ret.columns if c.startswith("G")]
    n_groups   = len(group_cols)
    has_ls     = "LS" in nav.columns

    # ── 颜色方案：tab10 定性色板，每组独立高辨识度颜色，LS 黑色 ──
    tab10  = plt.cm.tab10.colors
    colors = [tab10[i % 10] for i in range(n_groups)]

    # ── LS 绩效统计（用翻转后的 plot_ret，与图表显示一致）────────────
    if has_ls:
        ls_ret  = plot_ret["LS"].dropna()
        ann_ret = ls_ret.mean() * freq
        ann_vol = ls_ret.std() * np.sqrt(freq)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        ann_text = f"LS  Ann.Ret={ann_ret*100:.1f}%  Sharpe={sharpe:.2f}"
    else:
        ann_text = ""

    # ==========================================================
    # 模式 A：单图（单因子，dual_panel=False）
    # ==========================================================
    if not dual_panel:
        fig, ax = plt.subplots(figsize=(12, 6))

        # G1~Gn 分组线
        for i, col in enumerate(group_cols):
            ax.plot(nav.index, nav[col], color=colors[i],
                    linewidth=1.2, label=col)

        # LS 黑色粗虚线
        if has_ls:
            ax.plot(nav.index, nav["LS"], color="black", linewidth=2.0,
                    linestyle="--", label="LS")

        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(display_name, fontsize=14, pad=10)
        # 纵轴：NAV (log)
        # log scale → 纵轴等距 = 等比例涨跌（如 1→2 与 4→8 占同等高度）
        ax.set_ylabel("NAV (log)", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, _: f"{y:.1f}" if y < 10 else f"{int(y)}"
        ))
        ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

        # 右下角标注 LS 绩效
        if has_ls and ann_text:
            ax.text(0.98, 0.05, ann_text, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        fig.tight_layout()

    # ==========================================================
    # 模式 B：双子图（多因子合成，dual_panel=True）
    # ==========================================================
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 1.2]},
            sharex=True,
        )
        ax_grp, ax_ls = axes

        # 上图：分组 NAV
        for i, col in enumerate(group_cols):
            ax_grp.plot(nav.index, nav[col], color=colors[i],
                        linewidth=1.2, label=col)

        ax_grp.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax_grp.set_title(display_name, fontsize=14, pad=10)
        ax_grp.set_ylabel("NAV (log)", fontsize=10)
        ax_grp.set_yscale("log")
        ax_grp.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, _: f"{y:.1f}" if y < 10 else f"{int(y)}"
        ))
        ax_grp.legend(loc="upper left", fontsize=9, framealpha=0.7)
        ax_grp.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

        # 下图：LS 多空 NAV（线性坐标，合成因子 direction=1 LS 本已向上）
        if has_ls:
            ax_ls.plot(nav.index, nav["LS"], color="black", linewidth=1.8,
                       linestyle="--", label="LS (Long-Short)")
            ax_ls.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")

            if ann_text:
                ax_ls.text(0.98, 0.95, ann_text, transform=ax_ls.transAxes,
                           ha="right", va="top", fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

            ax_ls.set_ylabel("LS NAV", fontsize=10)
            ax_ls.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax_ls.legend(loc="upper left", fontsize=9, framealpha=0.7)
            ax_ls.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        else:
            ax_ls.set_visible(False)

        ax_ls.set_xlabel("Date", fontsize=10)
        fig.tight_layout(h_pad=0.5)

    # ── 保存（文件名用原始 factor_name，不用 display_name）────────────
    out_path = output_dir / f"{factor_name}_nav.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] 净值曲线已保存：{out_path.name}")
