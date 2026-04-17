# =============================================================================
# backtest/report.py
# 绩效报告生成
#
# TODO: 实现可视化报告输出（分组净值曲线、IC时序图、因子衰减图）
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from backtest.metrics import group_summary, ic_summary


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
