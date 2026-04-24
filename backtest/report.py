# =============================================================================
# backtest/report.py
# 绩效报告生成
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from backtest.metrics import group_summary, ic_summary

import matplotlib
matplotlib.use("Agg")          # 非交互后端，不弹窗，适合脚本/服务器运行
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


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


def plot_nav_curve(
    factor_name: str,
    group_ret: pd.DataFrame,
    output_dir: Path,
    freq: int = 12,
) -> None:
    """
    绘制分组累计净值曲线并保存为 PNG。

    图形内容
    --------
    - G1~Gn：各分组累计净值（tab10 多色，每组独立颜色）
    - LS：多空组合累计净值（黑色粗虚线，单独图例标注）
    - 右上角标注 LS 年化收益率与 Sharpe

    Parameters
    ----------
    factor_name : str
        因子名称，用于图标题和文件名
    group_ret : pd.DataFrame
        分组月度收益，index=月末日期，columns=[G1,...,Gn,LS]
        来自 engine.group_return()
    output_dir : Path
        图片输出目录（通常为 IMG_OUTPUT_DIR）
    freq : int
        数据年化频率（月度=12，季度=4），用于计算 Sharpe
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 累计净值（从 1 开始）──────────────────────────────────────
    nav = (1 + group_ret.fillna(0)).cumprod()

    group_cols = [c for c in group_ret.columns if c.startswith("G")]
    n_groups   = len(group_cols)

    # ── 颜色方案：tab10 定性色板，每组独立高辨识度颜色，LS 黑色 ──
    tab10  = plt.cm.tab10.colors
    colors = [tab10[i % 10] for i in range(n_groups)]

    # ── 双子图布局：上图 G1~Gn，下图 LS ────────────────────────────
    has_ls = "LS" in nav.columns
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1.2]},
        sharex=True,
    )
    ax_grp, ax_ls = axes

    # ── 上图：分组累计净值 ──────────────────────────────────────────
    for i, col in enumerate(group_cols):
        ax_grp.plot(nav.index, nav[col], color=colors[i],
                    linewidth=1.2, label=col)

    ax_grp.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_grp.set_title(f"{factor_name}  —  Cumulative Group NAV",
                     fontsize=14, pad=10)
    ax_grp.set_ylabel("Cumulative NAV (log)", fontsize=10)
    ax_grp.set_yscale("log")
    ax_grp.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{y:.1f}" if y < 10 else f"{int(y)}"
    ))
    ax_grp.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax_grp.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    # ── 下图：LS 多空组合 ───────────────────────────────────────────
    if has_ls:
        ax_ls.plot(nav.index, nav["LS"], color="black", linewidth=1.8,
                   linestyle="--", label="LS (Long-Short)")
        ax_ls.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")

        # 右上角注释：LS 年化收益 + Sharpe
        ls_ret  = group_ret["LS"].dropna()
        ann_ret = ls_ret.mean() * freq
        ann_vol = ls_ret.std() * np.sqrt(freq)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        ann_text = f"Ann.Ret={ann_ret*100:.1f}%  Sharpe={sharpe:.2f}"
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

    # ── 保存 ────────────────────────────────────────────────────────
    out_path = output_dir / f"{factor_name}_nav.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] 净值曲线已保存：{out_path.name}")
