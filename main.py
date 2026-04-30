# =============================================================================
# main.py
# 策略框架主入口
#
# 使用方式：
#   python main.py
#
# 控制逻辑：
#   通过 FACTOR_FLAGS 字典中的布林值控制哪些因子参与计算。
#   True  = 计算该因子并进行回测
#   False = 跳过
#
# 因子缓存：
#   因子值计算完成后自动保存至 factors/output/cache/<factor>.parquet
#   下次运行时直接读取缓存，跳过重复计算。
#   若需强制重新计算（如数据更新后），将 BACKTEST_CONFIG["force_recalc"] 设为 True。
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

from config.settings import HIST_START, FACTOR_OUTPUT_DIR, IMG_OUTPUT_DIR
from strategy.combine_factors import (
    align_factor_directions,
    calc_rolling_icir_weights,
    combine_factors,
    lowdin_orthogonalize,
)
from backtest.engine import calc_monthly_returns, group_return
from backtest.metrics import calc_ic
from backtest.report import print_factor_report, save_report, plot_nav_curve

# 因子缓存目录
CACHE_DIR = FACTOR_OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 因子计算开关（布林控制）
# 修改此处的 True/False 控制哪些因子参与计算和回测
# =============================================================================

# =============================================================================
# 多因子合成配置
# =============================================================================

# 各因子 IC 方向（+1 = 正向，-1 = 负向）
# 负向因子在合成时乘以 -1，使得高分 = 好股票
FACTOR_DIRECTIONS = {
    # 微观结构（已验证方向）
    "reversal_20":               -1,   # IC < 0，翻转
    "momentum_12_1":             -1,   # IC < 0，翻转
    "turnover_20":               -1,   # IC < 0，翻转
    "turnover_20_neutral":       -1,   # 同 turnover_20（待验证后确认）
    "amihud":                    +1,   # IC > 0，保持
    "amihud_neutral":            +1,   # 同 amihud（待验证后确认）
    "amihud_zero_adj":           +1,   # IC > 0，保持
    "amihud_zero_adj_neutral":   +1,   # 同 amihud_zero_adj（待验证后确认）
    "cs_spread":                 -1,   # IC < 0，翻转
    "roll_spread":               -1,   # IC < 0，翻转
    "overnight_ret":             +1,   # IC > 0，保持
    "volatility_30":             -1,   # IC < 0，翻转
    # 基本面（待回测后确认方向，暂按经济逻辑设定）
    "pb":                        -1,   # PB 越低 = 价值股，预期收益正向（低PB好）
    "bm":                        +1,   # B/M 越高 = 价值股，正向因子
    "pe_ttm":                    -1,   # PE 越低 = 价值股（低PE好）
    "pe1":                       -1,   # 同 pe_ttm
    "dividend_yield":            +1,   # 股息率越高越好
    "size":                      -1,   # 小市值溢价（小市值好）
    "size2":                     -1,   # 同 size
    "net_profit_yoy":            +1,   # 净利润增速越高越好
}

MULTI_FACTOR_CONFIG = {
    "run_combine":       False,  # ← 主开关：False = 只跑单因子，True = 同时跑多因子合成
    "use_orthogonalize": False,  # True = 先做 Lowdin 正交化再合成
    "icir_window":       12,     # 滚动 ICIR 权重的窗口（月）
    "run_equal_weight":  True,   # 是否运行等权版本
    "run_icir_weight":   True,   # 是否运行 ICIR 权重版本
}

# =============================================================================
# 多因子合成因子池
#
# 独立于 FACTOR_FLAGS：FACTOR_FLAGS 控制哪些因子跑单因子回测，
# COMBINE_FACTORS 控制哪些因子进入多因子合成，两者互不干扰。
#
# 使用说明：
#   - 同类双版本因子（如 turnover_20 / turnover_20_neutral）只选其一填入
#   - 将某一类全部注释掉 = 该类不参与合成（等效于分类开关）
#   - 列表中的因子必须在 FACTOR_FLAGS 中为 True（否则无缓存可用）
# =============================================================================

COMBINE_FACTORS = {
    "microstructure": [
        "turnover_20",       # 换手率（二选一：turnover_20 / turnover_20_neutral）
        "amihud",            # Amihud（二选一：amihud / amihud_neutral）
        "amihud_zero_adj",   # Amihud 零交易日调整版
        "reversal_20",       # 短期反转
        "momentum_12_1",     # 中期动量
        "cs_spread",         # CS 价差
        "roll_spread",       # Roll 价差
        "overnight_ret",     # 隔夜收益率
        "volatility_30",     # 短期波动率
    ],
    "fundamental": [
        # 单因子跑完对比后再填，示例：
        # "bm",              # 账面市值比（二选一：bm / pb）
        # "pe_ttm",          # 市盈率 TTM（二选一：pe_ttm / pe1）
        # "size",            # 市值（二选一：size / size2）
        # "dividend_yield",  # 股息率
    ],
}

FACTOR_FLAGS = {
    # ── 微观结构因子（活跃，可直接计算）────────────────────────────────────
    "reversal_20":               True,    # 短期反转（20日累计收益）
    "momentum_12_1":             True,    # 中期动量（12-1月）
    "turnover_20":               True,    # 换手率（20日均，无中性化）
    "turnover_20_neutral":       False,   # 换手率（20日均，流通市值中性化）
    "amihud":                    True,    # Amihud 非流动性（3月滚动，成交额口径，无中性化）
    "amihud_neutral":            False,   # Amihud 非流动性（3月滚动，流通市值中性化）
    "amihud_zero_adj":           True,    # Amihud 零交易日调整版（log+NT修正，无中性化）
    "amihud_zero_adj_neutral":   False,   # Amihud 零交易日调整版（流通市值中性化）
    "cs_spread":                 True,    # Corwin-Schultz 高低价价差
    "roll_spread":               True,    # Roll 价差
    "overnight_ret":             True,    # 隔夜收益率（月均）
    "volatility_30":             True,    # 短期波动率（30日）

    # ── 基本面因子（活跃，可直接计算）──────────────────────────────────────
    "pb":                        False,   # 市净率（无中性化）
    "bm":                        False,   # 账面市值比 log(1/PB)（无中性化，正向因子）
    "pe_ttm":                    False,   # 市盈率 TTM（无中性化）
    "pe1":                       False,   # 动态市盈率（无中性化）
    "dividend_yield":            False,   # 股息率 TTM（无中性化）
    "size":                      False,   # log(总市值)（无中性化）
    "size2":                     False,   # log(流通市值)（无中性化）
    "net_profit_yoy":            False,   # 净利润同比增速（流通市值+行业双重中性化）

    # ── 待激活因子（需补充数据后将 False 改为 True）─────────────────────────
    # [需补充数据] marketrtn_daily.csv（日度市场收益率序列）
    "ps_gamma":                  False,   # Pastor-Stambaugh Gamma

    # [需补充数据] ps_gamma 先激活 + marketrtn_daily.csv
    "ps_liq_beta":               False,   # PS 流动性 Beta（36月滚动）

    # [需补充数据] marketvalue.csv（日度流通市值序列）+ amt.csv（已有）
    "ap_betas":                  False,   # Acharya-Pedersen β1-β5

    # [需补充数据] marketrtn_daily.csv
    "capm_beta":                 False,   # CAPM 市场 Beta（240日滚动）

    # [需补充数据] FF3 日度因子（RiskPremium/HML/SMB）+ rf_daily.csv
    "ivol":                      False,   # 特质波动率（FF3 残差年化标准差）
    "ff3_betas":                 False,   # FF3 三因子 Beta
}

# =============================================================================
# 回测时间范围（在此处调整，其他地方不需要改）
#
# 可用数据范围参考：
#   存档（_archive/raw_data）：2014-01-01 ~ 2021-03-31
#     包含：后复权价格、换手率、PB、流通股本、ST标记、交易状态、上市天数
#   Database（/Mirror/MyProjects/Database）：2004-01-02 ~ 至今
#     包含：OHLCV、换手率、成交额、PE/PB/市值/股息率等全量指标
#
# 常用区间：
#   主回测   "2007-01-01" ~ None（全因子统一起点，PB/PE 数据可信度较好）
#   近十年   "2014-01-01" ~ "2024-12-31"（存档+Database 双源覆盖）
# =============================================================================

BACKTEST_START = "2007-01-01"   # ← 修改起始日期
#   选择 2007 起：2007 前 A 股市场规模小，PB/PE 数据可信度偏低，各因子统一对齐

# 截止日期：自动取上一个完整月末
#   逻辑：当月数据不完整，只用已完整收盘的月份
#   例：今天 2026-04-21 → 自动设为 2026-03-31
#   每次运行自动更新，无需手动修改
BACKTEST_END = str(
    (pd.Timestamp.today().to_period("M") - 1).to_timestamp("M").date()
)

# =============================================================================
# 回测参数
# =============================================================================

BACKTEST_CONFIG = {
    "start":        BACKTEST_START,
    "end":          BACKTEST_END,
    "n_groups":     5,             # 分组数（5 或 10）
    "freq":         12,            # 数据频率（月度=12，季度=4）
    "save_output":  True,          # 是否保存回测结果到 FACTOR_OUTPUT_DIR
    "force_recalc": False,         # True = 忽略缓存、强制重新计算所有因子
                                   # （数据更新后或修改因子逻辑后使用）
}

# =============================================================================
# 因子计算函数映射
# =============================================================================

def _get_factor_func(factor_name: str):
    """根据因子名称返回对应的计算函数。"""
    from factors.microstructure import (
        calc_reversal_20, calc_momentum_12_1,
        calc_turnover_20, calc_turnover_20_neutral,
        calc_amihud, calc_amihud_neutral,
        calc_amihud_zero_adj, calc_amihud_zero_adj_neutral,
        calc_cs_spread, calc_roll_spread,
        calc_overnight_ret, calc_volatility_30,
        _calc_ps_gamma, _calc_ps_liq_beta, _calc_ap_betas,
        _calc_capm_beta, _calc_ivol, _calc_ff3_betas,
    )
    from factors.fundamental import (
        calc_pb, calc_bm, calc_pe_ttm, calc_pe1,
        calc_dividend_yield, calc_size, calc_size2, calc_net_profit_yoy,
    )

    mapping = {
        # 微观结构（无中性化）
        "reversal_20":               calc_reversal_20,
        "momentum_12_1":             calc_momentum_12_1,
        "turnover_20":               calc_turnover_20,
        "amihud":                    calc_amihud,
        "amihud_zero_adj":           calc_amihud_zero_adj,
        "cs_spread":                 calc_cs_spread,
        "roll_spread":               calc_roll_spread,
        "overnight_ret":             calc_overnight_ret,
        "volatility_30":             calc_volatility_30,
        # 微观结构（流通市值中性化）
        "turnover_20_neutral":       calc_turnover_20_neutral,
        "amihud_neutral":            calc_amihud_neutral,
        "amihud_zero_adj_neutral":   calc_amihud_zero_adj_neutral,
        # 待激活
        "ps_gamma":                  _calc_ps_gamma,
        "ps_liq_beta":               _calc_ps_liq_beta,
        "ap_betas":                  _calc_ap_betas,
        "capm_beta":                 _calc_capm_beta,
        "ivol":                      _calc_ivol,
        "ff3_betas":                 _calc_ff3_betas,
        # 基本面
        "pb":                        calc_pb,
        "bm":                        calc_bm,
        "pe_ttm":                    calc_pe_ttm,
        "pe1":                       calc_pe1,
        "dividend_yield":            calc_dividend_yield,
        "size":                      calc_size,
        "size2":                     calc_size2,
        "net_profit_yoy":            calc_net_profit_yoy,
    }
    return mapping.get(factor_name)


# =============================================================================
# 主流程
# =============================================================================

def _load_factor_cached(
    factor_name: str,
    func,
    start: str,
    end,
    force_recalc: bool,
) -> pd.DataFrame | None:
    """
    因子加载（带缓存）。

    缓存策略：
      - 缓存文件：CACHE_DIR/<factor_name>.csv（仅依赖 pandas，无需 pyarrow）
      - 命中缓存且 force_recalc=False → 直接读取，跳过计算
      - 未命中或 force_recalc=True   → 重新计算并写入缓存
    """
    cache_path = CACHE_DIR / f"{factor_name}.csv"

    if not force_recalc and cache_path.exists():
        print(f"  ✓ 读取缓存：{cache_path.name}")
        df = pd.read_csv(cache_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

    # 计算因子
    factor = func(start=start, end=end)
    if factor is None or factor.empty:
        return None

    # 写入缓存
    factor.to_csv(cache_path)
    action = "重新计算并缓存" if force_recalc and cache_path.exists() else "计算完成，已缓存"
    print(f"  ✓ {action}：{cache_path.name}  shape={factor.shape}")
    return factor


def main():
    start        = BACKTEST_CONFIG["start"]
    end          = BACKTEST_CONFIG["end"]
    n_groups     = BACKTEST_CONFIG["n_groups"]
    freq         = BACKTEST_CONFIG["freq"]
    save_out     = BACKTEST_CONFIG["save_output"]
    force_recalc = BACKTEST_CONFIG.get("force_recalc", False)

    print("=" * 60)
    print("  量化因子回测框架")
    print(f"  回测区间：{start} ~ {end or '至今'}")
    print(f"  分组数：{n_groups}，频率：{'月度' if freq == 12 else '季度'}")
    print(f"  因子缓存：{'强制重算' if force_recalc else '启用（命中则跳过计算）'}")
    print("=" * 60)

    # --- 步骤 1：计算月度收益率（全局复用）---
    print("\n[Step 1] 计算月度持仓收益率...")
    try:
        monthly_ret = calc_monthly_returns(start=start, end=end)
        print(f"  ✓ 月度收益率矩阵：{monthly_ret.shape}")
    except Exception as e:
        print(f"  ✗ 月度收益率计算失败：{e}")
        return

    # --- 步骤 2：逐因子加载（缓存优先）+ 回测 ---
    active_factors = [name for name, flag in FACTOR_FLAGS.items() if flag]
    print(f"\n[Step 2] 共 {len(active_factors)} 个因子待处理：{active_factors}")

    results = {}

    for factor_name in active_factors:
        print(f"\n{'─' * 40}")
        print(f"  因子：{factor_name}")

        func = _get_factor_func(factor_name)
        if func is None:
            print(f"  ✗ 未找到因子函数：{factor_name}")
            continue

        # 因子加载（缓存优先）
        try:
            factor = _load_factor_cached(
                factor_name, func, start, end, force_recalc
            )
            if factor is None or factor.empty:
                print(f"  ✗ {factor_name} 返回空 DataFrame，跳过")
                continue
        except NotImplementedError as e:
            print(f"  ⚠ {factor_name} 尚未激活：{e}")
            continue
        except Exception as e:
            warnings.warn(f"  ✗ {factor_name} 计算异常：{e}")
            continue

        # 分组回测
        # fwd_ret：下期收益（T月末因子 → 预测 T+1月收益，避免前瞻偏差）
        try:
            fwd_ret   = monthly_ret.shift(-1)
            grp_ret   = group_return(factor, fwd_ret, n_groups=n_groups)
            ic_series = calc_ic(factor, fwd_ret, method="spearman")

            results[factor_name] = {
                "factor":    factor,
                "group_ret": grp_ret,
                "ic_series": ic_series,
            }

            print_factor_report(factor_name, grp_ret, ic_series, freq=freq)

            if save_out:
                save_report(factor_name, grp_ret, ic_series, FACTOR_OUTPUT_DIR, freq=freq)
                plot_nav_curve(factor_name, grp_ret, IMG_OUTPUT_DIR, freq=freq)

        except Exception as e:
            warnings.warn(f"  ✗ {factor_name} 回测异常：{e}")
            continue

    # --- 步骤 3：单因子汇总 ---
    print(f"\n{'=' * 60}")
    print(f"  单因子完成！成功处理 {len(results)}/{len(active_factors)} 个因子")
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

    # --- 步骤 4：多因子合成打分回测 ---
    if not MULTI_FACTOR_CONFIG.get("run_combine", False):
        print(f"\n{'=' * 60}")
        print("  [Step 4] 多因子合成：已关闭（run_combine=False）")
        print("  单因子回测完成。如需运行合成，将 run_combine 改为 True。")
        print("=" * 60)
        return results

    print(f"\n{'=' * 60}")
    print("  [Step 4] 多因子合成打分模型")
    print("=" * 60)

    # 4.1 从 COMBINE_FACTORS 中收集参与合成的因子
    combine_names = [f for lst in COMBINE_FACTORS.values() for f in lst]

    # 校验：必须在 results 中（已跑过单因子）且在 FACTOR_DIRECTIONS 中
    missing = [n for n in combine_names if n not in results]
    if missing:
        print(f"  ⚠ 以下因子不在单因子结果中（未在 FACTOR_FLAGS 中激活或计算失败）：{missing}")
        combine_names = [n for n in combine_names if n in results]

    factors_for_combine = {
        name: results[name]["factor"]
        for name in combine_names
        if name in FACTOR_DIRECTIONS
    }
    active_directions = {
        name: FACTOR_DIRECTIONS[name]
        for name in factors_for_combine
    }

    if len(factors_for_combine) < 2:
        print("  ⚠ 可用因子数不足 2 个，跳过多因子合成")
        return results

    micro_in = [f for f in COMBINE_FACTORS.get("microstructure", []) if f in factors_for_combine]
    fund_in  = [f for f in COMBINE_FACTORS.get("fundamental", [])    if f in factors_for_combine]
    print(f"\n  参与合成因子（共 {len(factors_for_combine)} 个）")
    if micro_in:
        print(f"    微观结构（{len(micro_in)}）：{micro_in}")
    if fund_in:
        print(f"    基本面  （{len(fund_in)}）：{fund_in}")

    # 4.2 方向对齐
    print("\n  方向对齐...")
    aligned = align_factor_directions(factors_for_combine, active_directions)

    # 4.3 可选：Lowdin 正交化
    if MULTI_FACTOR_CONFIG.get("use_orthogonalize", False):
        print("\n  Lowdin 正交化...")
        aligned = lowdin_orthogonalize(aligned)

    fwd_ret  = monthly_ret.shift(-1)

    # 4.4 等权合成
    if MULTI_FACTOR_CONFIG.get("run_equal_weight", True):
        print("\n" + "─" * 40)
        print("  多因子合成：等权（Equal Weight）")
        composite_eq = combine_factors(aligned, weights="equal")
        grp_ret_eq   = group_return(composite_eq, fwd_ret, n_groups=n_groups)
        ic_eq        = calc_ic(composite_eq, fwd_ret, method="spearman")

        print_factor_report("multi_factor_equal", grp_ret_eq, ic_eq, freq=freq)
        if save_out:
            save_report("multi_factor_equal", grp_ret_eq, ic_eq,
                        FACTOR_OUTPUT_DIR, freq=freq)
            plot_nav_curve("multi_factor_equal", grp_ret_eq,
                           IMG_OUTPUT_DIR, freq=freq)

    # 4.5 ICIR 权重合成
    if MULTI_FACTOR_CONFIG.get("run_icir_weight", True):
        print("\n" + "─" * 40)
        print("  多因子合成：滚动 ICIR 权重")
        print(f"  滚动窗口：{MULTI_FACTOR_CONFIG['icir_window']} 个月")

        icir_weights = calc_rolling_icir_weights(
            aligned, monthly_ret,
            window=MULTI_FACTOR_CONFIG["icir_window"],
        )
        print("\n  最新一期各因子权重：")
        latest_w = icir_weights.dropna(how="all").iloc[-1]
        for fname, w in latest_w.sort_values(ascending=False).items():
            print(f"    {fname}: {w:.4f}")

        composite_ir = combine_factors(aligned, weights=icir_weights)
        grp_ret_ir   = group_return(composite_ir, fwd_ret, n_groups=n_groups)
        ic_ir        = calc_ic(composite_ir, fwd_ret, method="spearman")

        print_factor_report("multi_factor_icir", grp_ret_ir, ic_ir, freq=freq)
        if save_out:
            save_report("multi_factor_icir", grp_ret_ir, ic_ir,
                        FACTOR_OUTPUT_DIR, freq=freq)
            plot_nav_curve("multi_factor_icir", grp_ret_ir,
                           IMG_OUTPUT_DIR, freq=freq)

    print(f"\n{'=' * 60}")
    print("  多因子合成完成！")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
