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
#   下次运行时先做新鲜度校验，通过才读缓存，否则自动重算并覆盖。
#   校验两条判据（见 _cache_stale_reason）：
#     ① 缓存文件 mtime 必须晚于 Database 源目录的最新 mtime
#        → Database 跑过 update.py / 换过数据源，缓存立即失效
#     ② 缓存末行必须覆盖到本次 BACKTEST_END 所在月末
#        → 跨月后缓存自动失效
#   因此常规使用保持 force_recalc=False 即可，不必手工判断该不该重算。
#   只有「改了因子计算逻辑本身」（Database 数据没动，mtime 不变）才需手工置 True。
#
# 待激活因子（标注 [需补充数据]）：
#   这些因子代码已实现，但需要在 Database 中补充对应数据后才能启用。
#   补充数据后将 False 改为 True 即可激活。
# =============================================================================

import contextlib
import sys
import time
import warnings
import functools
import pandas as pd
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import (
    HIST_START,
    FACTOR_OUTPUT_DIR,
    DATABASE_DIR,
    HK_DATABASE_DIR,
)
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


def _as_list(x):
    """把 int 或 list 统一成 list。n_groups 兼容两种写法。"""
    return list(x) if isinstance(x, (list, tuple)) else [x]


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

RUN_A  = True     # ← A 股回测开关
RUN_HK = False    # ← 港股回测开关
                  #   2026-08-23 关闭：港股数据冻结于 2026-07-17
                  #   （Database/src/update.py 自 2026-08-17 起注释掉 US/HK 更新），
                  #   月度回测最后一期不完整，结果不可用。
                  #   恢复港股更新后再置回 True。

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
    "pe1":                       -1,   # ⚠️ 现为 Choice PE（静态口径），非原动态 PE
    "dividend_yield":            +1,   # 股息率越高越好
    "ps_ttm":                    -1,   # PS 越低 = 越便宜
    "ev_ebitda":                 -1,   # 企业倍数越低 = 越便宜
    "est_pe_ftm":                -1,   # 预测 PE 越低 = 越便宜
    "est_peg":                   +1,   # ⚠️ 实测方向与直觉相反：高 PEG 反而跑赢，见 CLAUDE.md
    "ev2_neutral":               -1,   # 同等规模下企业价值越低 = 越便宜（先按逻辑设，跑完回填）
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
    "pe1":                       True,   # 市盈率（⚠️ 现为 Choice PE 静态口径）
    "dividend_yield":            True,   # 股息率 TTM（无中性化）
    "ps_ttm":                    True,   # 市销率 TTM（无中性化）
    "ev_ebitda":                 True,   # 企业倍数 EV2/EBITDA（无中性化）
    "est_pe_ftm":                True,   # 预测 PE 未来12月（无中性化）⚠️ 覆盖约 51%
    "est_peg":                   True,   # 预测 PEG（无中性化）⚠️ 覆盖约 51%
    "ev2_neutral":               True,   # log(EV2)，流通市值中性化
    "size":                      True,   # log(总市值)（无中性化）
    "size2":                     True,   # log(流通市值)（无中性化）
    "net_profit_yoy":            False,   # 净利润同比增速（流通市值+行业双重中性化）

    # ── 风险/流动性风险因子 ────────────────────────────────────────────────
    # [已补充] FF3 日度因子 + rf_daily.csv（2026-07-15 就绪）
    "ivol":                      True,    # 特质波动率（FF3 残差年化标准差）
    "ff3_betas":                 True,    # FF3 市场 Beta（= β_mkt，口径未变）

    # ★ 2026-08-18 激活。原标注「需补 marketrtn_daily.csv」已失效：
    #   ff3_daily.csv 的 MKT 是超额市场收益，r_m = MKT + Rf 即所需序列；
    #   ap_betas 需要的日度流通市值就是 neg_market_value.csv。均已就绪。
    "beta_smb":                  True,    # FF3 规模载荷（原塞在 attrs 里被丢弃）
    "beta_hml":                  True,    # FF3 价值载荷（同上）
    "capm_beta":                 True,    # CAPM 市场 Beta（240日滚动）
    "ps_gamma":                  True,    # Pastor-Stambaugh Gamma
    "ps_liq_beta":               True,    # PS 流动性 Beta（36月滚动）
    "ap_beta1":                  True,    # Acharya-Pedersen β1（市场收益）
    "ap_beta2":                  True,    # AP β2（流动性共动）
    "ap_beta3":                  True,    # AP β3（收益 vs 市场流动性）
    "ap_beta4":                  True,    # AP β4（流动性 vs 市场收益）
    "ap_beta5":                  True,    # AP β5 = β2 − β3 − β4（净流动性）

    # ── ★ 2026-08-18 特征扩充（src/factors/expansion.py，22 个）───────────
    # 为 ML 面板准备特征宽度。入选标准：GKX/JKP 有清晰构建方法 + Choice
    # 现有字段可构建 + **不做估值反推**（财报数据优先，没有就不做）。
    # 动量 / 反转族
    "ret_2_1":                   True,
    "ret_3_1":                   True,
    "ret_6_1":                   True,
    "ret_6_0":                   True,
    "ret_9_1":                   True,
    "ret_12_7":                  True,
    "ret_36_13":                 True,   # 长期反转
    "seasonality":               True,   # 季节性动量（6~10年同月）
    "prc_high_252":              True,   # 52周高点接近度
    # 波动 / 极值族
    "rvol_21":                   True,
    "rvol_252":                  True,
    "rmax1_21":                  True,   # MAX effect
    "rmax5_21":                  True,
    "skew_21":                   True,
    # 流动性 / 微观结构族
    # ✅ 2026-08-20 重构并重新启用：改为 Liu (2006) 的 LM 指标
    #    （252日窗口 + 换手率打破平局）。初版「21日零成交占比」的分组塌陷
    #    不是数据性质而是构建缺陷——漏了 Liu 原文里专门用来打破平局的
    #    换手率项，导致截面 96%~99.7% 取同一个值。修正后各截面取值几乎全不同
    #    （5192 只股票 5191 个不同取值）。详见 expansion.py 该函数注释。
    "zero_trades_252":           True,
    "dolvol_126":                True,
    "turn_std_21":               True,
    "close_vwap_dev":            True,   # ★ VWAP 独有
    "amihud_vwap":               True,   # ★ VWAP 独有
    "high_low_range":            True,
    # 规模 / 股本族
    "free_float_ratio":          True,   # ★ A股特色
    "float_shares_chg":          True,   # 解禁压力
    "age":                       True,
    # 估值比率的时序变换（仅变换，无反推）
    "ep":                        True,
    "sp":                        True,
    "pb_chg_12":                 True,
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
    "n_groups":     5,             # 分组数。可以是 int，也可以是 list
                                   #   5        → 只跑五分组（默认，与全部历史记录一致）
                                   #   [5, 10]  → 五分组 + 十分组各出一份报告
                                   # ⚠️ **列表的第一个是主口径**：它的输出文件不带后缀、
                                   #    并用于多因子合成；其余只额外产出 `_gN` 后缀的报告。
                                   #    默认保持 5 不动——CLAUDE.md / FACTORS.md 里
                                   #    62 个因子的全部结果与验收基准都是五分组的。
                                   #    十分组是 GKX/JKP 的学术惯例，与 src/ml 对比时用。
    "freq":         12,            # 数据频率（月度=12，季度=4）
    "save_output":  True,          # 是否保存回测结果到 FACTOR_OUTPUT_DIR
    "force_recalc": False,         # True = 无条件重算所有因子，忽略新鲜度校验
                                   # False = 走新鲜度校验（推荐，见文件头「因子缓存」）
                                   #   数据更新 / 跨月 会被自动识别并重算，
                                   #   只有改了因子计算逻辑本身才需手工置 True。
                                   # 注：2026-08 A 股数据源 iFinD→Choice 的全量重算
                                   #     已于 2026-08-23 完成，缓存均已基于 Choice。
}

# =============================================================================
# 因子计算函数映射
# =============================================================================

def _get_factor_func(factor_name: str):
    """根据因子名称返回对应的计算函数。"""
    from src.factors.microstructure import (
        calc_reversal_20, calc_momentum_12_1,
        calc_turnover_20, calc_turnover_20_neutral,
        calc_amihud, calc_amihud_neutral,
        calc_amihud_zero_adj, calc_amihud_zero_adj_neutral,
        calc_cs_spread, calc_roll_spread,
        calc_overnight_ret, calc_volatility_30,
        _calc_ps_gamma, _calc_ps_liq_beta,
        _calc_ap_beta1, _calc_ap_beta2, _calc_ap_beta3,
        _calc_ap_beta4, _calc_ap_beta5,
        _calc_capm_beta, _calc_ivol, _calc_ff3_betas,
        _calc_beta_smb, _calc_beta_hml,
    )
    from src.factors.fundamental import (
        calc_pb, calc_bm, calc_pe_ttm, calc_pe1,
        calc_dividend_yield, calc_size, calc_size2, calc_net_profit_yoy,
        calc_ps_ttm, calc_ev_ebitda, calc_est_pe_ftm, calc_est_peg,
        calc_ev2_neutral,
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
        # 风险 / 流动性风险
        "ps_gamma":                  _calc_ps_gamma,
        "ps_liq_beta":               _calc_ps_liq_beta,
        "ap_beta1":                  _calc_ap_beta1,
        "ap_beta2":                  _calc_ap_beta2,
        "ap_beta3":                  _calc_ap_beta3,
        "ap_beta4":                  _calc_ap_beta4,
        "ap_beta5":                  _calc_ap_beta5,
        "capm_beta":                 _calc_capm_beta,
        "ivol":                      _calc_ivol,
        "ff3_betas":                 _calc_ff3_betas,
        "beta_smb":                  _calc_beta_smb,
        "beta_hml":                  _calc_beta_hml,
        # 基本面
        "pb":                        calc_pb,
        "bm":                        calc_bm,
        "pe_ttm":                    calc_pe_ttm,
        "pe1":                       calc_pe1,
        "dividend_yield":            calc_dividend_yield,
        "size":                      calc_size,
        "size2":                     calc_size2,
        "net_profit_yoy":            calc_net_profit_yoy,
        "ps_ttm":                    calc_ps_ttm,
        "ev_ebitda":                 calc_ev_ebitda,
        "est_pe_ftm":                calc_est_pe_ftm,
        "est_peg":                   calc_est_peg,
        "ev2_neutral":               calc_ev2_neutral,
    }

    # ── 特征扩充（expansion.py）：函数名统一为 calc_<因子名> ──────────────
    from src.factors import expansion
    for _name in (
        "ret_2_1", "ret_3_1", "ret_6_1", "ret_6_0", "ret_9_1", "ret_12_7",
        "ret_36_13", "seasonality", "prc_high_252",
        "rvol_21", "rvol_252", "rmax1_21", "rmax5_21", "skew_21",
        "zero_trades_252", "dolvol_126", "turn_std_21",
        "close_vwap_dev", "amihud_vwap", "high_low_range",
        "free_float_ratio", "float_shares_chg", "age",
        "ep", "sp", "pb_chg_12",
    ):
        mapping[_name] = getattr(expansion, f"calc_{_name}")

    return mapping.get(factor_name)


# =============================================================================
# 阶段计时
# =============================================================================
#
# 起因：实测「62 个因子全部命中缓存」仍要约 11 分钟，而读完 65 个缓存文件
# （878 MB）只需 6 秒——说明耗时既不在因子计算、也不在缓存 IO，而在因子算完
# **之后**那一段。但那一段里有两个嫌疑人（分组回测的 pandas 运算 vs
# matplotlib 画图），在分清楚之前任何优化都是猜。
#
# 这里只做测量，不做优化：累计各阶段耗时，Step 2 结束时打印一张分布表。
# 开销是每次调用一次 perf_counter，可忽略。

_STAGE_SECS: dict[str, float] = {}


@contextlib.contextmanager
def _stage(name: str):
    """累计某个阶段的耗时。嵌套安全（各自独立计各自的）。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _STAGE_SECS[name] = _STAGE_SECS.get(name, 0.0) + time.perf_counter() - t0


def _print_stage_breakdown() -> None:
    """打印阶段耗时分布。无数据时静默跳过。"""
    if not _STAGE_SECS:
        return
    total = sum(_STAGE_SECS.values())
    if total <= 0:
        return
    print(f"\n{'─' * 60}")
    print(f"  阶段耗时分布（合计 {total:.1f}s）")
    print(f"{'─' * 60}")
    for name, secs in sorted(_STAGE_SECS.items(), key=lambda kv: -kv[1]):
        bar = "█" * max(1, round(secs / total * 30))
        print(f"    {name:<18s} {secs:7.1f}s  {secs / total * 100:5.1f}%  {bar}")
    print(f"{'─' * 60}")


# =============================================================================
# 主流程
# =============================================================================

@functools.lru_cache(maxsize=4)
def _database_mtime(market: str) -> float:
    """
    Database 源目录下所有 CSV 的最新 mtime。

    整轮回测只 stat 一次（lru_cache），62 个因子共用同一个值。
    取整个目录的 max 而非单看 close.csv：choice_derive.py --promote 是
    28 个文件一起拷入，但 listed_days.csv 由 choice_listed_days.py 单独
    生效，晚几秒；且基本面因子读的是 pb/pe_ttm 而非 close。取 max 才不漏。
    """
    src_dir = HK_DATABASE_DIR if market == "HK" else DATABASE_DIR
    return max((f.stat().st_mtime for f in src_dir.glob("*.csv")), default=0.0)


def _cache_stale_reason(cache_path: Path, end, market: str) -> str | None:
    """
    缓存新鲜度校验。返回 None = 可用；返回字符串 = 失效原因（用于打印）。

    判据 1  Database 数据比缓存新
            → update.py 跑过、或数据源整体重建过，缓存基于旧数据，必须重算。
    判据 2  缓存末行没覆盖到本次 BACKTEST_END 所在月末
            → 跨月了，缓存短一期。

    任一不满足即失效。校验失败一律返回原因而非抛错——最坏情况是多算一次，
    不会因为缓存文件损坏让整轮回测挂掉。
    """
    try:
        if cache_path.stat().st_mtime < _database_mtime(market):
            return "Database 数据已更新"
    except OSError:
        return "缓存文件不可读"

    # 只读第一列（日期索引），避免为了看一个日期把 15MB 宽表整个载入
    try:
        idx = pd.read_csv(cache_path, index_col=0, usecols=[0]).index
        if len(idx) == 0:
            return "缓存为空表"
        last = pd.Timestamp(idx[-1])
    except Exception:
        return "缓存无法解析"

    need = pd.Timestamp(end).to_period("M").to_timestamp("M")
    if last < need:
        return f"缓存末行 {last.date()} 未覆盖回测截止 {need.date()}"

    return None


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
      - 命中缓存 + force_recalc=False + 通过新鲜度校验 → 直接读取，跳过计算
      - 未命中 / force_recalc=True / 校验不通过        → 重新计算并覆盖缓存
      新鲜度校验见 _cache_stale_reason（数据更新、跨月均会自动失效）。
    """
    cache_path = CACHE_DIR / f"{cache_prefix}{factor_name}.csv"
    market = "HK" if cache_prefix == "hk_" else "A"

    if not force_recalc and cache_path.exists():
        stale = _cache_stale_reason(cache_path, end, market)
        if stale is None:
            print(f"  ✓ 读取缓存：{cache_path.name}")
            df = pd.read_csv(cache_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df
        print(f"  ⟳ 缓存失效（{stale}），重算：{cache_path.name}")

    # 计算因子
    factor = func(start=start, end=end)
    if factor is None or factor.empty:
        return None

    # 写入缓存。
    # existed 必须在 to_csv **之前**取：写完之后 exists() 恒为 True，
    # 原实现在写盘后才判断，于是「失效重算」被打印成「计算完成，已缓存」，
    # 与首次计算长得一模一样——排查「缓存到底失效没有」时日志会误导人。
    existed = cache_path.exists()
    factor.to_csv(cache_path)
    if not existed:
        action = "计算完成，已缓存"
    elif force_recalc:
        action = "强制重算并覆盖缓存"
    else:
        action = "失效重算并覆盖缓存"
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
    n_groups,          # int 或 list[int]
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
    _gs = _as_list(n_groups)
    print(f"  分组数：{_gs[0]}（主口径）"
          + (f" + {_gs[1:]}（附加报告）" if len(_gs) > 1 else "")
          + f"，频率：{'月度' if freq == 12 else '季度'}")
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
            with _stage("因子加载/计算"):
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
            with _stage("calc_ic"):
                ic_series = calc_ic(factor, fwd_ret, method="spearman")

            # n_groups 可以是 int 或 list。**首个分组数是主口径**：
            # 它的结果进 results（供多因子合成用）、输出文件不带后缀，
            # 与历史文件名和文档里的记录完全一致。
            # 其余分组数只额外产出一份带 `_gN` 后缀的报告，互不干扰。
            for gi, ng in enumerate(_as_list(n_groups)):
                with _stage("group_return"):
                    grp_ret = group_return(factor, fwd_ret, n_groups=ng)
                suffix  = "" if gi == 0 else f"_g{ng}"

                if gi == 0:
                    results[factor_name] = {
                        "factor":    factor,
                        "group_ret": grp_ret,
                        "ic_series": ic_series,
                    }

                with _stage("print_report"):
                    print_factor_report(f"{factor_name}{suffix}", grp_ret,
                                        ic_series, freq=freq)

                if save_out:
                    out_name = f"{factor_name}{suffix}"
                    with _stage("save_report"):
                        save_report(out_name, grp_ret, ic_series, stats_dir,
                                    freq=freq)
                    factor_direction = directions.get(factor_name, 1)
                    with _stage("plot_nav_curve"):
                        plot_nav_curve(out_name, grp_ret, img_dir, freq=freq,
                                       direction=factor_direction)

        except Exception as e:
            warnings.warn(f"  ✗ {factor_name} 回测异常：{e}")
            continue

    # --- 步骤 3：单因子汇总 ---
    _print_stage_breakdown()
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
        grp_ret_eq   = group_return(composite_eq, fwd_ret, n_groups=_as_list(n_groups)[0])
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
        grp_ret_ir   = group_return(composite_ir, fwd_ret, n_groups=_as_list(n_groups)[0])
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
