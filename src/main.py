# =============================================================================
# main.py
# 策略框架主入口
#
# 使用方式：
#   python -m src.main   （以包方式运行，须在 Quant/ 根目录执行）
#
# 控制逻辑：
#   通过 FACTOR_FLAGS 字典中的布林值控制哪些因子参与计算。
#   True  = 计算该因子并进行回测
#   False = 跳过
#
# 因子缓存：
#   因子值计算完成后自动保存至 output/cache/<factor>（缓存目录）
#   下次运行时直接读取缓存，跳过重复计算。
#   若需强制重新计算（如数据更新后），将 BACKTEST_CONFIG["force_recalc"] 设为 True。
#
# 待激活因子（标注 [需补充数据]）：
#   这些因子代码已实现，但需要在 Database 中补充对应数据后才能启用。
#   补充数据后将 False 改为 True 即可激活。
# =============================================================================

import sys
import warnings
import functools
import pandas as pd
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import HIST_START, FACTOR_OUTPUT_DIR
from src.strategy.combine_factors import (
    align_factor_directions,
    calc_rolling_icir_weights,
    combine_factors,
    lowdin_orthogonalize,
)
from src.backtest.engine import calc_monthly_returns, group_return
from src.backtest.metrics import calc_ic
from src.backtest.report import print_factor_report, save_report, plot_nav_curve

# 因子缓存目录
CACHE_DIR = FACTOR_OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 市场回测开关（独立控制，互不影响）
# =============================================================================
#
# 每个市场均可独立开关：
#   RUN_A  = True  → 运行 A 股回测
#   RUN_HK = True  → 运行港股回测
#   两者都 True    → 先跑 A 股，再跑港股（各自独立缓存，互不干扰）
#   两者都 False   → 跳过所有回测（打印提示后退出）
#
# 港股可用因子（受限于现有数据，见 FACTOR_FLAGS_HK）：
#   turnover_20  换手率（turn.csv）
#   amihud       Amihud 非流动性（close_adj + amt）
#   cs_spread    CS 价差（high_adj + low_adj）
#
# 港股不可用因子（数据不存在，仅 A 股专属）：
#   所有基本面因子（pe/pb/size/bm/dividend_yield 等）
#   reversal/momentum/roll_spread/overnight_ret/volatility（依赖存档或不复权价格）
# =============================================================================

RUN_A  = True    # ← A 股回测开关
RUN_HK = True    # ← 港股回测开关

# -----------------------------------------------------------------------------
# 港股因子计算开关（仅 RUN_HK=True 时生效）
# -----------------------------------------------------------------------------
FACTOR_FLAGS_HK = {
    "turnover_20": True,   # 换手率（20日均）
    "amihud":      True,   # Amihud 非流动性（3月滚动）
    "cs_spread":   True,   # Corwin-Schultz 高低价价差
}

# -----------------------------------------------------------------------------
# 港股多因子合成配置（仅 RUN_HK=True 时生效）
# -----------------------------------------------------------------------------
COMBINE_FACTORS_HK = {
    "microstructure": ["turnover_20", "amihud", "cs_spread"],
}

FACTOR_DIRECTIONS_HK = {
    "turnover_20": -1,   # 换手率越低 → 预期收益越高（参考 A 股结论）
    "amihud":      +1,   # 非流动性越高 → 流动性溢价，预期收益越高
    "cs_spread":   -1,   # 价差越低（流动性越好）→ 预期收益越高
}

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
    "run_combine":       True,   # ← 主开关：False = 只跑单因子，True = 同时跑多因子合成
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
    # ── 微观结构（7个）────────────────────────────────────────────────────────
    # 因子选择依据（2007-2026 单因子回测 + 截面 Spearman 相关性分析）：
    #   · turnover_20_neutral 替代 turnover_20（2026-05 数据源统一后更新）：
    #     数据源切换至纯 iFinD 后，原始 turn 与市值强相关，ICIR 从 -0.715 降至 -0.383；
    #     市值中性化版本 ICIR -0.712，LS Sharpe 1.35，信号纯净且与 size 低相关，
    #     合成中已有 size，故用中性化版本剥离冗余。
    #   · amihud vs amihud_zero_adj → amihud ICIR 更强（0.380 vs 0.265），选 amihud
    #   · momentum_12_1 ICIR=-0.099，A股失效，不纳入
    "microstructure": [
        "turnover_20_neutral",  # ICIR -0.712 ★★★ 换手率（市值中性化，纯流动性信号）
        "reversal_20",          # ICIR -0.450 ★★★ 短期反转
        "roll_spread",          # ICIR -0.481 ★★★ Roll 价差
        "cs_spread",            # ICIR -0.438 ★★★ CS 价差
        "amihud",               # ICIR +0.380 ★★★ Amihud 非流动性（无中性化）
        "volatility_30",        # ICIR -0.377 ★★  短期波动率
        "overnight_ret",        # ICIR +0.219 ★   隔夜收益率（弱但独立信号）
    ],
    # ── 基本面（3个）────────────────────────────────────────────────────────
    # 因子选择依据：
    #   · bm vs pb → bm 正向，符合 Fama-French HML 逻辑，选 bm
    #   · pe1 vs pe_ttm → pe1 更及时（最新季报年化），ICIR -0.298 > pe_ttm -0.241
    #   · size vs size2 → size（总市值）ICIR -0.355 > size2 -0.263
    #   · size ↔ amihud 截面相关 ρ=-0.770，信息有重叠，但经济含义不同，保留两者（方案A）
    #   · dividend_yield LS Sharpe 异常（1.91 vs ICIR 0.181），混入规模效应，暂不纳入
    "fundamental": [
        "size",           # ICIR -0.355 ★★★ 小市值溢价（总市值）
        "bm",             # ICIR +0.334 ★★  账面市值比（价值因子，Fama-French HML）
        "pe1",            # ICIR -0.298 ★★  动态市盈率（最新季报年化）
    ],
}

FACTOR_FLAGS = {
    # ── 微观结构因子（活跃，可直接计算）────────────────────────────────────
    "reversal_20":               True,    # 短期反转（20日累计收益）
    "momentum_12_1":             True,    # 中期动量（12-1月）
    "turnover_20":               True,    # 换手率（20日均，无中性化）
    "turnover_20_neutral":       True,   # 换手率（20日均，流通市值中性化）
    "amihud":                    True,    # Amihud 非流动性（3月滚动，成交额口径，无中性化）
    "amihud_neutral":            True,   # Amihud 非流动性（3月滚动，流通市值中性化）
    "amihud_zero_adj":           True,    # Amihud 零交易日调整版（log+NT修正，无中性化）
    "amihud_zero_adj_neutral":   True,   # Amihud 零交易日调整版（流通市值中性化）
    "cs_spread":                 True,    # Corwin-Schultz 高低价价差
    "roll_spread":               True,    # Roll 价差
    "overnight_ret":             True,    # 隔夜收益率（月均）
    "volatility_30":             True,    # 短期波动率（30日）

    # ── 基本面因子（活跃，可直接计算）──────────────────────────────────────
    "pb":                        True,   # 市净率（无中性化）
    "bm":                        True,   # 账面市值比 log(1/PB)（无中性化，正向因子）
    "pe_ttm":                    True,   # 市盈率 TTM（无中性化）
    "pe1":                       True,   # 动态市盈率（无中性化）
    "dividend_yield":            True,   # 股息率 TTM（无中性化）
    "size":                      True,   # log(总市值)（无中性化）
    "size2":                     True,   # log(流通市值)（无中性化）
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

    # [已补充] FF3 日度因子 + rf_daily.csv（2026-07-15 就绪）
    "ivol":                      True,    # 特质波动率（FF3 残差年化标准差）
    "ff3_betas":                 True,    # FF3 三因子 Beta
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
    "force_recalc": True,          # True = 忽略缓存、强制重新计算所有因子
                                   # （数据更新后或修改因子逻辑后使用）
                                   # ⚠️ 2026-08 A 股数据源换成 Choice，close_adj 的复权
                                   #    基准与 iFinD 不同（000001.SZ 一类整条序列平移 16%），
                                   #    旧缓存全部失效，必须置 True 跑一轮
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
    cache_prefix: str = "",
) -> pd.DataFrame | None:
    """
    因子加载（带缓存）。

    缓存策略：
      - 缓存文件：CACHE_DIR/<cache_prefix><factor_name>.csv
      - A 股（默认）：cache_prefix=""，文件名如 "turnover_20.csv"
      - 港股：       cache_prefix="hk_"，文件名如 "hk_turnover_20.csv"
      - 命中缓存且 force_recalc=False → 直接读取，跳过计算
      - 未命中或 force_recalc=True   → 重新计算并写入缓存
    """
    cache_path = CACHE_DIR / f"{cache_prefix}{factor_name}.csv"

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


def _run_single_market(
    market: str,
    market_label: str,
    active_flags: dict,
    combine_cfg: dict,
    directions: dict,
    cache_prefix: str,
    start: str,
    end,
    n_groups: int,
    freq: int,
    save_out: bool,
    force_recalc: bool,
) -> dict:
    """
    单市场完整回测流程（单因子 + 多因子合成）。

    由 main() 调用，按市场拆开后各自独立运行。
    返回 {factor_name: {"factor": ..., "group_ret": ..., "ic_series": ...}}
    """
    print("=" * 60)
    print(f"  量化因子回测框架  [{market_label}]")
    print(f"  回测区间：{start} ~ {end or '至今'}")
    print(f"  分组数：{n_groups}，频率：{'月度' if freq == 12 else '季度'}")
    print(f"  因子缓存：{'强制重算' if force_recalc else '启用（命中则跳过计算）'}")
    print("=" * 60)

    # 市场专属输出目录（按 market 动态构建，自动创建）
    stats_dir = FACTOR_OUTPUT_DIR / market / "stats"
    img_dir   = FACTOR_OUTPUT_DIR / market / "img"
    stats_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # --- 步骤 1：计算月度收益率（全局复用）---
    print(f"\n[Step 1] 计算月度持仓收益率（{market_label}）...")
    try:
        monthly_ret = calc_monthly_returns(start=start, end=end, market=market)
        print(f"  ✓ 月度收益率矩阵：{monthly_ret.shape}")
    except Exception as e:
        print(f"  ✗ 月度收益率计算失败：{e}")
        return {}

    # --- 步骤 2：逐因子加载（缓存优先）+ 回测 ---
    active_factors = [name for name, flag in active_flags.items() if flag]
    print(f"\n[Step 2] 共 {len(active_factors)} 个因子待处理：{active_factors}")

    results = {}

    for factor_name in active_factors:
        print(f"\n{'─' * 40}")
        print(f"  因子：{factor_name}")

        func = _get_factor_func(factor_name)
        if func is None:
            print(f"  ✗ 未找到因子函数：{factor_name}")
            continue

        # 非 A 股市场：将 market 绑定到因子函数（partial 不影响 A 股默认调用）
        if market != "A":
            func = functools.partial(func, market=market)

        # 因子加载（缓存优先）
        try:
            factor = _load_factor_cached(
                factor_name, func, start, end, force_recalc,
                cache_prefix=cache_prefix,
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
                out_name = f"{factor_name}"
                save_report(out_name, grp_ret, ic_series, stats_dir, freq=freq)
                factor_direction = directions.get(factor_name, 1)
                plot_nav_curve(out_name, grp_ret, img_dir, freq=freq,
                               direction=factor_direction)

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

    # 4.1 从 combine_cfg 中收集参与合成的因子
    combine_names = [f for lst in combine_cfg.values() for f in lst]

    # 校验：必须在 results 中（已跑过单因子）且在 directions 中
    missing = [n for n in combine_names if n not in results]
    if missing:
        print(f"  ⚠ 以下因子不在单因子结果中（未在 FACTOR_FLAGS 中激活或计算失败）：{missing}")
        combine_names = [n for n in combine_names if n in results]

    factors_for_combine = {
        name: results[name]["factor"]
        for name in combine_names
        if name in directions
    }
    active_directions = {
        name: directions[name]
        for name in factors_for_combine
    }

    if len(factors_for_combine) < 2:
        print("  ⚠ 可用因子数不足 2 个，跳过多因子合成")
        return results

    micro_in = [f for f in combine_cfg.get("microstructure", []) if f in factors_for_combine]
    fund_in  = [f for f in combine_cfg.get("fundamental",    []) if f in factors_for_combine]
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

    fwd_ret = monthly_ret.shift(-1)

    # 4.4 等权合成
    if MULTI_FACTOR_CONFIG.get("run_equal_weight", True):
        print("\n" + "─" * 40)
        print("  多因子合成：等权（Equal Weight）")
        composite_eq = combine_factors(aligned, weights="equal")
        grp_ret_eq   = group_return(composite_eq, fwd_ret, n_groups=n_groups)
        ic_eq        = calc_ic(composite_eq, fwd_ret, method="spearman")

        eq_name = "multi_factor_equal"
        print_factor_report(eq_name, grp_ret_eq, ic_eq, freq=freq)
        if save_out:
            save_report(eq_name, grp_ret_eq, ic_eq, stats_dir, freq=freq)
            plot_nav_curve(eq_name, grp_ret_eq, img_dir, freq=freq, dual_panel=True)

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

        ir_name = "multi_factor_icir"
        print_factor_report(ir_name, grp_ret_ir, ic_ir, freq=freq)
        if save_out:
            save_report(ir_name, grp_ret_ir, ic_ir, stats_dir, freq=freq)
            plot_nav_curve(ir_name, grp_ret_ir, img_dir, freq=freq, dual_panel=True)

    print(f"\n{'=' * 60}")
    print("  多因子合成完成！")
    print("=" * 60)

    return results


def main():
    """
    主入口：按 RUN_A / RUN_HK 开关依次运行各市场回测。

    开关组合说明：
        RUN_A=True,  RUN_HK=False → 只跑 A 股（默认）
        RUN_A=False, RUN_HK=True  → 只跑港股（研究港股时 A 股不计算）
        RUN_A=True,  RUN_HK=True  → A 股 + 港股依次运行
        RUN_A=False, RUN_HK=False → 不运行任何回测（打印提示后退出）
    """
    start        = BACKTEST_CONFIG["start"]
    end          = BACKTEST_CONFIG["end"]
    n_groups     = BACKTEST_CONFIG["n_groups"]
    freq         = BACKTEST_CONFIG["freq"]
    save_out     = BACKTEST_CONFIG["save_output"]
    force_recalc = BACKTEST_CONFIG.get("force_recalc", False)

    # 构建待运行市场列表（保持固定顺序：A 股 → 港股）
    markets_to_run = []
    if RUN_A:
        markets_to_run.append(dict(
            market       = "A",
            market_label = "A 股",
            active_flags = FACTOR_FLAGS,
            combine_cfg  = COMBINE_FACTORS,
            directions   = FACTOR_DIRECTIONS,
            cache_prefix = "",
        ))
    if RUN_HK:
        markets_to_run.append(dict(
            market       = "HK",
            market_label = "港股（HK）",
            active_flags = FACTOR_FLAGS_HK,
            combine_cfg  = COMBINE_FACTORS_HK,
            directions   = FACTOR_DIRECTIONS_HK,
            cache_prefix = "hk_",
        ))

    if not markets_to_run:
        print("⚠  RUN_A 和 RUN_HK 均为 False，无市场需要回测，退出。")
        return {}

    all_results = {}
    for mkt in markets_to_run:
        all_results[mkt["market"]] = _run_single_market(
            market       = mkt["market"],
            market_label = mkt["market_label"],
            active_flags = mkt["active_flags"],
            combine_cfg  = mkt["combine_cfg"],
            directions   = mkt["directions"],
            cache_prefix = mkt["cache_prefix"],
            start        = start,
            end          = end,
            n_groups     = n_groups,
            freq         = freq,
            save_out     = save_out,
            force_recalc = force_recalc,
        )

    return all_results


if __name__ == "__main__":
    main()
