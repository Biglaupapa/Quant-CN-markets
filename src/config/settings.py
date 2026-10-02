# =============================================================================
# config/settings.py
# 全局路径与参数配置
# =============================================================================

from pathlib import Path

# -----------------------------------------------------------------------------
# 项目根目录（本文件位于 src/config/settings.py，故向上三级到 Quant 根）
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent

# -----------------------------------------------------------------------------
# 数据路径
# -----------------------------------------------------------------------------

# 历史存档数据（原始 CSV，2014-2020）
ARCHIVE_DATA_DIR = PROJECT_ROOT / "_archive" / "raw_data"

# Database：iFinD/Wind 实时更新数据（2021-至今）
DATABASE_DIR = Path("/Users/louis/MyProjects/Database/data/stock/A")

# Database：港股日度数据（iFinD，2004-01-02 至今）
HK_DATABASE_DIR = Path("/Users/louis/MyProjects/Database/data/stock/HK")

# 行业（用于行业中性化）：聚源 dz_exgindustry「申万行业分类(新)」月末宽表，时点口径，2005 起全覆盖。
# 2026-10-02 替代 Wind h5（_archive/raw_data/FactorLoading_Industry_arch.h5：止于 2021-02、
# 覆盖 76%~97%，且因缺 pytables + 日期格式不匹配从未生效）。生成：jydb_extract.py --what industry_sw
INDUSTRY_PATH = Path("/Users/louis/MyProjects/Database/data/vendor/jydb/industry_sw_jydb.csv")

# 回测结果根目录（与 factors/、backtest/ 等模块平级）
# 实际输出按市场分层：
#   output/A/stats/   → A 股 CSV 统计文件
#   output/A/img/     → A 股净值曲线图
#   output/HK/stats/  → 港股 CSV 统计文件
#   output/HK/img/    → 港股净值曲线图
#   output/cache/     → 因子缓存（共享，不按市场分）
FACTOR_OUTPUT_DIR = PROJECT_ROOT / "output"
FACTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AF-pricing 因子方法参考路径（不复制代码，仅作来源注释）
AF_PRICING_CODING_DIR = Path("/Users/louis/MyProjects/AF-pricing/Coding")

# -----------------------------------------------------------------------------
# 港股 Database 字段映射
# 数据来源：iFinD，格式 index=日期, columns=股票代码（与 A 股 Database 相同）
# 覆盖：close_adj / open_adj / high_adj / low_adj / volume / amt / turn
# 注意：港股无 A 股的 ST / status / listed_days / PE / PB 等，
#       Universe 过滤仅依赖 close_adj（见 data/universe.py build_investable_mask_hk）
# -----------------------------------------------------------------------------
HK_DATABASE_FIELD_MAP = {
    "close_adj": "close_adj.csv",
    "open_adj":  "open_adj.csv",
    "high_adj":  "high_adj.csv",
    "low_adj":   "low_adj.csv",
    "volume":    "volume.csv",
    "amt":       "amt.csv",
    "turn":      "turn.csv",
}

# 港股仙股过滤：股价低于此值（HKD）的股票剔除（等价于 A 股 ST 剔除）
PENNY_STOCK_PRICE_MIN = 1.0   # HKD

# -----------------------------------------------------------------------------
# 存档数据字段映射
# 原始 CSV 格式：index=股票代码, columns=日期（转置格式）
#
# !! 重要说明（2026-05 修订）!!
# Archive 来自 Wind，Database 来自 iFinD + Datayes，两个来源定义和复权算法不同。
# Database 已覆盖 2003 年至今的完整历史（6077 只股票），因此：
#   - 除 net_profit 外，所有字段均应使用 Database，不再混用 Archive。
#   - 混用会导致同一字段在不同时间段来自不同数据源，造成拼接断层。
#   - net_profit 是唯一的 Archive 专属字段（Database 无净利润数据）。
# -----------------------------------------------------------------------------
ARCHIVE_FIELD_MAP = {
    "net_profit":     "归属母公司净利润.csv",   # 归母净利润（季度），Archive 专属
}

# -----------------------------------------------------------------------------
# Database 字段映射
# Database CSV 格式：index=日期, columns=股票代码（标准格式）
# -----------------------------------------------------------------------------
DATABASE_FIELD_MAP = {
    # 价格（不复权）—— 2026-10-02 起权威来源为财汇 tq_qt_skdailyprice 直接字段（第②级），
    # 替代 Choice 派生（MV/TOTALSHARE，第③级）。全时段单一来源、不拼接；财汇缺格保持 NaN。
    # 见 Database/docs/【登记】指标权威来源.md。生成：python3 src/vendor_caihui_quote.py
    "close":           "../../vendor/caihui/wide/close_caihui.csv",
    "open":            "../../vendor/caihui/wide/open_caihui.csv",
    "high":            "../../vendor/caihui/wide/high_caihui.csv",
    "low":             "../../vendor/caihui/wide/low_caihui.csv",
    "high_adj":        "high_adj.csv",
    "low_adj":         "low_adj.csv",
    # 后复权价格（原为存档专属，现已补入 Database）
    "close_adj":       "close_adj.csv",
    "open_adj":        "open_adj.csv",
    # 交易量
    "volume":          "volume.csv",
    "amt":             "amt.csv",
    "turn":            "turn.csv",
    # 自由流通换手率（聚源 dz_stockperformance.TurnoverRateFreeFloat，%，日度；2026-09-29 接入）
    # 分母 = 自由流通股本（流通股再剔除大股东/高管等长期持股），与 turn 同形、停牌日已置 NaN。
    # 文件在 Database 的供应商隔离区，不在 data/stock/A/，故用相对路径跳出去。
    # 生成：cd ~/MyProjects/Database && python3 src/jydb_extract.py --what turn_ff（需 VPN）
    "turn_ff":         "../../vendor/jydb/turn_ff_jydb.csv",
    # 成交均价 VWAP（= AMOUNT/VOLUME，Choice 全精度；2026-08-18 接入）
    # ★ 这是相对 CRSP / JKP 的独有字段——美国学术库不提供日内 VWAP，
    #   故 close/vwap 偏离、VWAP 口径 Amihud 这类特征在英文文献里罕见。
    "vwap":            "vwap.csv",             # 不复权
    "vwap_adj":        "vwap_adj.csv",         # 后复权
    # 基本面估值（快照）
    "pe_ttm":           "pe_ttm.csv",           # 市盈率 TTM（Choice PETTM）
    # ⚠️ 键名 pe1 是历史遗留，文件其实是 Choice 的 PE（最近年报静态市盈率）。
    #    原 Datayes pe1 = 市值/(最新单季净利×4)，是单季年化动态 PE；Choice 26 个字段
    #    里没有净利润，重建不出该口径（与 PE 秩相关仅 0.20~0.43），故于 2026-08-17
    #    改用 PE 顶替并沿用旧因子名。这是语义替换，该因子历史值已整体改变。
    "pe1":              "pe.csv",               # Choice PE（静态市盈率）
    # 市盈率（最新报告期年化、扣非；财汇 PEMRQNPAAEI，按公告日更新）→ ep_lsy、CH-3 VMG（2026-10-02）
    "pe_mrq_deducted":  "../../vendor/caihui/wide/pe_mrq_deducted_caihui.csv",
    "pb":               "pb.csv",               # 市净率（Choice PB）
    "dividend_ratio":   "dividend_ratio.csv",   # 股息率（Choice LASTESTDIVIDEND，%）
    "ps_ttm":           "ps_ttm.csv",           # 市销率 TTM（Choice PSTTM）
    "ev2":              "ev2.csv",              # 企业价值(剔除货币资金)，单位：元
    "ev_ebitda":        "ev_ebitda.csv",        # 企业倍数 EV2/EBITDA
    "est_pe_ftm":       "est_pe_ftm.csv",       # 预测市盈率（未来12月），覆盖约 51%
    "est_peg":          "est_peg.csv",          # 预测 PEG，覆盖约 51%
    # 市值（三个相似但不同的指标，用途见登记表）
    "market_value":     "market_value.csv",     # 总市值（含 H/B 股，按 A 股价；Choice MV）→ size、市值分母因子
    # 流通市值：2026-10-02 起财汇 tq_sk_finindic.NEGOTIABLEMV（第②级），替代 Choice 派生 D3
    "neg_market_value": "../../vendor/caihui/wide/neg_market_value_caihui.csv",  # → 中性化、size2、FF3
    # A 股市值（含限售股）：财汇 tq_sk_finindic.TOTMKTCAP（第②级）→ 规则 g、LSY 复现、CH-3/CH-4
    "a_market_value":   "../../vendor/caihui/wide/a_market_value_caihui.csv",
    # 股票池过滤（原为存档专属，现已补入 Database）
    # Database 中字段名与存档不同，此处统一映射
    "is_st":            "st.csv",               # 1=ST，0=正常
    "trade_status":     "status.csv",           # 1=正常交易，0=停牌（与存档字符串格式不同）
    "listing_days":     "listed_days.csv",      # 上市至今交易日数
    "float_shares":     "float_shares.csv",     # 流通股本（Choice LIQSHARE；2026-10-02 由 free_float_shares 改名）
    "total_shares":     "total_shares.csv",     # 总股本（含 H/B 股，Choice TOTALSHARE；2026-10-02 新增）
}

# -----------------------------------------------------------------------------
# 时间范围
# -----------------------------------------------------------------------------
HIST_START = "2014-01-01"   # 历史 CSV 数据起始（用于存档 adjusted 价格）
HIST_END   = "2021-03-31"   # 历史 CSV 数据截止（存档实际到 2021-03-31）
DB_START   = "2005-01-04"   # Database 数据起始
# ⚠️ 2026-08 数据源变更：A 股已 100% 迁至 Choice，Datayes 于 2026-08-17 完全退场。
#    起点由 2003-01-02 改为 2005-01-04，股票列由 6077 改为 5861（时点并集）。
#    当前形状 5250 × 5861，覆盖 2005-01-04 ~ 2026-08-14，全部 29 个字段口径一致。
#    以 .sources.json 为准，不要以本注释为准——核实方法是直接读各 CSV 末行日期。
#    详见 Database/docs/【主文档】框架全景记录与Choice迁移方案.md

# -----------------------------------------------------------------------------
# 股票池过滤参数
# -----------------------------------------------------------------------------
IPO_FILTER_DAYS          = 60    # 新股过滤：上市不足 N 个交易日的股票排除
MIN_ROLLING_VALID_DAYS   = 10    # 滚动窗口内最少有效交易日（用于因子计算）
# ↑ 以上两项只用于**逐日**清洗掩码 build_investable_mask(freq="D")（因子计算输入）。
#   组建日（t 月末）的股票池由下面的 FORMATION_CONFIG 决定（2026-10-01 起与文献对齐）。

# -----------------------------------------------------------------------------
# 组建日股票池（t 月末）—— 与 LSY 2019 / HQZ / 顾明等对齐，见 docs/【方法】回测口径与文献对齐.md
# -----------------------------------------------------------------------------
# 规则 → 依据：
#   sample              a  样本：lsy = 60/00/30；lsy_star = + 科创板 688/689；all = + 北交所     LSY
#   trading_on_formation b  t 月最后交易日有成交（status=1）                                       HQZ
#   min_trade_days_month c  t 月成交天数 ≥ 15                                                       LSY
#   min_trade_days_12m   c  过去 12 个月成交天数 ≥ 120                                              LSY
#   min_listed_months    d  上市满 N 个月（日历，list_date + N 月 ≤ t）                              LSY、HQZ
#   exclude_st           e  t 月最后交易日 ST / *ST 剔除                                            顾明等
#   exclude_delist_period f t 月末处于待退市（退市整理期）剔除                                      顾明等
#   exclude_bottom_size  g  t 月末 A 股市值最小 bottom_size_pct 剔除（布尔开关）                   LSY、HQZ
#   bottom_size_pct         剔除比例，默认 0.30
#                           市值 = a_market_value（财汇 tq_sk_finindic.TOTMKTCAP，A 股市值含限售股，
#                           与 LSY 口径一致；2026-10-02 Louis 决定）
#                           排序范围：所选样本内当月有市值的全部股票
#                           ★ 主框架默认关闭：被剔除的小市值股票的收益同样重要（2026-10-02 Louis 决定）；
#                             LSY 复现、CH-3/CH-4 构建时显式打开
FORMATION_CONFIG = {
    "sample":                "lsy",
    "trading_on_formation":  True,
    "min_trade_days_month":  15,
    "min_trade_days_12m":    120,
    "min_listed_months":     6,
    "exclude_st":            True,
    "exclude_delist_period": True,
    "exclude_bottom_size":   False,
    "bottom_size_pct":       0.30,
}

# 组建日股票池所需的 Database 侧数据
STOCKS_LIST_PATH    = Path("/Users/louis/MyProjects/Database/data/stocks_list/A.csv")          # 证券主表（list_date）
DELIST_PERIOD_PATH  = Path("/Users/louis/MyProjects/Database/data/stock/A_monthly/delist_period.csv")   # 待退市月末名单（Choice sector 001026）

# -----------------------------------------------------------------------------
# 因子计算参数
# -----------------------------------------------------------------------------

# 微观结构因子
REVERSAL_WINDOW          = 20    # 短期反转因子回看窗口（交易日）
MOMENTUM_LONG            = 12    # 中期动量因子长端（月）
MOMENTUM_SKIP            = 1     # 中期动量因子跳过最近 N 月（避免短期反转）
TURNOVER_WINDOW          = 20    # 换手率滚动窗口（交易日）
AMIHUD_WINDOW_MONTHS     = 3     # Amihud 滚动窗口（月）
AMIHUD_LAG_MONTHS        = 1     # Amihud 滞后月数（避免前瞻偏差）
ILLIQ_SCALE              = 1e5   # Amihud 缩放系数（成交额已换算为百万元后使用）
VOLATILITY_WINDOW        = 30    # 短期波动率窗口（交易日）

# 需要额外数据的复杂因子（FF3 / 市场收益率序列）
CAPM_BETA_WINDOW         = 240   # CAPM Beta 滚动窗口（交易日，≈12个月）
IVOL_WINDOW_MONTHS       = 12    # 特质波动率 FF3 回归窗口（月）
PS_GAMMA_MIN_OBS         = 10    # Pastor-Stambaugh Gamma 最小月内观测数
PS_BETA_WINDOW_MONTHS    = 36    # PS Liquidity Beta 滚动窗口（月）
AP_BETA_WINDOW_MONTHS    = 36    # Acharya-Pedersen Beta 滚动窗口（月）
AP_C1, AP_C2, AP_C3      = 0.25, 0.30, 60.0  # AP illiquidity cost 参数

# 基本面因子
PB_NEUTRALIZE_SIZE       = True  # PB 是否做市值中性化
PB_NEUTRALIZE_INDUSTRY   = True  # PB 是否做行业中性化
NP_YOY_LAG_QUARTERS      = 3     # 净利润同比增速滞后季度数（财报披露延迟）

# 换手率中性化
TURN_NEUTRALIZE_SIZE     = True  # 换手率做市值中性化

# -----------------------------------------------------------------------------
# 价格复权选择（影响所有基于价格收益率的因子）
# True  = 后复权：close_adj / open_adj（默认，避免除权日收益失真）
# False = 不复权：close / open（如需对比或复现特定文献时使用）
#
# 不受此开关影响的因子（有固定原因）：
#   amihud     → 始终用 close_adj（日度收益率必须复权）
#   cs_spread  → 始终用 high_adj / low_adj（跨日价格比较必须复权）
#   turnover_20→ 市值计算用 close（不复权，代表当日真实市值）
# -----------------------------------------------------------------------------
USE_ADJ_PRICE            = True  # ← 修改此处切换价格来源
