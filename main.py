# =============================================================================
# main.py
# 策略框架主入口
#
# 使用方式：
#   python main.py
#
# 控制逻辑：
#   在 main() 函数中通过 FACTOR_FLAGS 字典中的布林值控制哪些因子参与计算。
#   True  = 计算该因子并进行回测
#   False = 跳过
#
# 待激活因子（标注 [需补充数据]）：
#   这些因子代码已实现，但需要在 Database 中补充对应数据后才能启用。
#   补充数据后将 False 改为 True 即可激活。
# =============================================================================

import sys
import warnings
import pandas as pd
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import HIST_START, FACTOR_OUTPUT_DIR
from backtest.engine import calc_monthly_returns, group_return
from backtest.metrics import calc_ic
from backtest.report import print_factor_report, save_report


# =============================================================================
# 因子计算开关（布林控制）
# 修改此处的 True/False 控制哪些因子参与计算和回测
# =============================================================================

FACTOR_FLAGS = {
    # ── 微观结构因子（活跃，可直接计算）────────────────────────────────────
    "reversal_20":       True,   # 短期反转（20日累计收益）
    "momentum_12_1":     True,   # 中期动量（12-1月）
    "turnover_20":       True,   # 换手率（20日均，市值中性化）
    "amihud":            True,   # Amihud 非流动性（版本A，3月滚动，成交额口径）
    "amihud_zero_adj":   True,   # Amihud 零交易日调整版（版本C，log+NT修正）
    "cs_spread":         True,   # Corwin-Schultz 高低价价差
    "roll_spread":       True,   # Roll 价差
    "overnight_ret":     True,   # 隔夜收益率（月均）
    "volatility_30":     True,   # 短期波动率（30日）

    # ── 基本面因子（活跃，可直接计算）──────────────────────────────────────
    "pb":                True,   # 市净率（市值+行业双重中性化）
    "pe_ttm":            True,   # 市盈率 TTM（仅 Database 2021+）
    "dividend_yield":    True,   # 股息率 TTM（仅 Database 2021+）
    "size":              True,   # 市值因子 log(流通市值)
    "net_profit_yoy":    True,   # 净利润同比增速（市值+行业双重中性化）

    # ── 待激活因子（需补充数据后将 False 改为 True）─────────────────────────
    # [需补充数据] marketrtn_daily.csv（日度市场收益率序列）
    "ps_gamma":          False,  # Pastor-Stambaugh Gamma

    # [需补充数据] ps_gamma 先激活 + marketrtn_daily.csv
    "ps_liq_beta":       False,  # PS 流动性 Beta（36月滚动）

    # [需补充数据] marketvalue.csv（日度流通市值序列）+ amt.csv（已有）
    "ap_betas":          False,  # Acharya-Pedersen β1-β5

    # [需补充数据] marketrtn_daily.csv
    "capm_beta":         False,  # CAPM 市场 Beta（240日滚动）

    # [需补充数据] FF3 日度因子（RiskPremium/HML/SMB）+ rf_daily.csv
    "ivol":              False,  # 特质波动率（FF3 残差年化标准差）
    "ff3_betas":         False,  # FF3 三因子 Beta
}

# =============================================================================
# 回测参数
# =============================================================================

BACKTEST_CONFIG = {
    "start":    "2021-04-01",   # 回测起始（Database 有数据后）
    "end":      None,           # None = 至今
    "n_groups": 5,              # 分组数（5 或 10）
    "freq":     12,             # 数据频率（月度=12，季度=4）
    "save_output": True,        # 是否保存结果到 FACTOR_OUTPUT_DIR
}

# =============================================================================
# 因子计算函数映射
# =============================================================================

def _get_factor_func(factor_name: str):
    """根据因子名称返回对应的计算函数。"""
    from factors.microstructure import (
        calc_reversal_20, calc_momentum_12_1, calc_turnover_20,
        calc_amihud, calc_amihud_zero_adj, calc_cs_spread,
        calc_roll_spread, calc_overnight_ret, calc_volatility_30,
        _calc_ps_gamma, _calc_ps_liq_beta, _calc_ap_betas,
        _calc_capm_beta, _calc_ivol, _calc_ff3_betas,
    )
    from factors.fundamental import (
        calc_pb, calc_pe_ttm, calc_dividend_yield,
        calc_size, calc_net_profit_yoy,
    )

    mapping = {
        # 微观结构
        "reversal_20":     calc_reversal_20,
        "momentum_12_1":   calc_momentum_12_1,
        "turnover_20":     calc_turnover_20,
        "amihud":          calc_amihud,
        "amihud_zero_adj": calc_amihud_zero_adj,
        "cs_spread":       calc_cs_spread,
        "roll_spread":     calc_roll_spread,
        "overnight_ret":   calc_overnight_ret,
        "volatility_30":   calc_volatility_30,
        # 待激活
        "ps_gamma":        _calc_ps_gamma,
        "ps_liq_beta":     _calc_ps_liq_beta,
        "ap_betas":        _calc_ap_betas,
        "capm_beta":       _calc_capm_beta,
        "ivol":            _calc_ivol,
        "ff3_betas":       _calc_ff3_betas,
        # 基本面
        "pb":              calc_pb,
        "pe_ttm":          calc_pe_ttm,
        "dividend_yield":  calc_dividend_yield,
        "size":            calc_size,
        "net_profit_yoy":  calc_net_profit_yoy,
    }
    return mapping.get(factor_name)


# =============================================================================
# 主流程
# =============================================================================

def main():
    start    = BACKTEST_CONFIG["start"]
    end      = BACKTEST_CONFIG["end"]
    n_groups = BACKTEST_CONFIG["n_groups"]
    freq     = BACKTEST_CONFIG["freq"]
    save_out = BACKTEST_CONFIG["save_output"]

    print("=" * 60)
    print("  量化因子回测框架")
    print(f"  回测区间：{start} ~ {end or '至今'}")
    print(f"  分组数：{n_groups}，频率：{'月度' if freq == 12 else '季度'}")
    print("=" * 60)

    # --- 步骤 1：计算月度收益率（全局复用）---
    print("\n[Step 1] 计算月度持仓收益率...")
    try:
        monthly_ret = calc_monthly_returns(start=start, end=end)
        print(f"  ✓ 月度收益率矩阵：{monthly_ret.shape}")
    except Exception as e:
        print(f"  ✗ 月度收益率计算失败：{e}")
        return

    # --- 步骤 2：逐因子计算 + 回测 ---
    active_factors = [name for name, flag in FACTOR_FLAGS.items() if flag]
    print(f"\n[Step 2] 共 {len(active_factors)} 个因子待计算：{active_factors}")

    results = {}

    for factor_name in active_factors:
        print(f"\n{'─' * 40}")
        print(f"  计算因子：{factor_name}")

        func = _get_factor_func(factor_name)
        if func is None:
            print(f"  ✗ 未找到因子函数：{factor_name}")
            continue

        # 因子计算
        try:
            factor = func(start=start, end=end)
            if factor is None or factor.empty:
                print(f"  ✗ {factor_name} 返回空 DataFrame，跳过")
                continue
            print(f"  ✓ 因子形状：{factor.shape}")
        except NotImplementedError as e:
            print(f"  ⚠ {factor_name} 尚未激活：{e}")
            continue
        except Exception as e:
            warnings.warn(f"  ✗ {factor_name} 计算异常：{e}")
            continue

        # 分组回测
        try:
            grp_ret  = group_return(factor, monthly_ret, n_groups=n_groups)
            # 下期收益（forward return = 当期因子对应下一期收益）
            fwd_ret  = monthly_ret.shift(-1)
            ic_series = calc_ic(factor, fwd_ret, method="spearman")

            results[factor_name] = {
                "factor":    factor,
                "group_ret": grp_ret,
                "ic_series": ic_series,
            }

            # 输出报告
            print_factor_report(factor_name, grp_ret, ic_series, freq=freq)

            if save_out:
                save_report(factor_name, grp_ret, ic_series, FACTOR_OUTPUT_DIR, freq=freq)

        except Exception as e:
            warnings.warn(f"  ✗ {factor_name} 回测异常：{e}")
            continue

    # --- 步骤 3：汇总 ---
    print(f"\n{'=' * 60}")
    print(f"  完成！成功计算 {len(results)}/{len(active_factors)} 个因子")
    if results:
        ic_means = {
            name: res["ic_series"].mean()
            for name, res in results.items()
            if not res["ic_series"].dropna().empty
        }
        ic_df = pd.Series(ic_means).sort_values(ascending=False)
        print("\n  各因子 RankIC 均值排名：")
        print(ic_df.to_string())
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
