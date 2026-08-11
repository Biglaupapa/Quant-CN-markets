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

# 行业因子载荷（HDF5，用于行业中性化，来自原始框架）
INDUSTRY_H5_PATH = ARCHIVE_DATA_DIR / "FactorLoading_Industry_arch.h5"

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
    # 价格（不复权）
    "close":           "close.csv",
    "open":            "open.csv",
    "high":            "high.csv",
    "low":             "low.csv",
    "high_adj":        "high_adj.csv",
    "low_adj":         "low_adj.csv",
    # 后复权价格（原为存档专属，现已补入 Database）
    "close_adj":       "close_adj.csv",
    "open_adj":        "open_adj.csv",
    # 交易量
    "volume":          "volume.csv",
    "amt":             "amt.csv",
    "turn":            "turn.csv",
    # 基本面估值（快照）
    "pe_ttm":           "pe_ttm.csv",           # 市盈率 TTM（静态，基于最新年报）
    "pe1":              "pe1.csv",              # 动态市盈率（滚动12个月盈利预测）
    "pb":               "pb.csv",               # 市净率
    "dividend_ratio":   "dividend_ratio.csv",   # 股息率（近12个月，%）
    # 市值
    "market_value":     "market_value.csv",     # 总市值（元）→ Size 因子
    "neg_market_value": "neg_market_value.csv", # 流通市值（元）→ Size2 因子
    # 股票池过滤（原为存档专属，现已补入 Database）
    # Database 中字段名与存档不同，此处统一映射
    "is_st":            "st.csv",               # 1=ST，0=正常
    "trade_status":     "status.csv",           # 1=正常交易，0=停牌（与存档字符串格式不同）
    "listing_days":     "listed_days.csv",      # 上市至今交易日数
    "float_shares":     "free_float_shares.csv", # 流通股本（股）
}

# -----------------------------------------------------------------------------
# 时间范围
# -----------------------------------------------------------------------------
HIST_START = "2014-01-01"   # 历史 CSV 数据起始（用于存档 adjusted 价格）
HIST_END   = "2021-03-31"   # 历史 CSV 数据截止（存档实际到 2021-03-31）
DB_START   = "2005-01-04"   # Database 数据起始
# ⚠️ 2026-08 数据源变更：A 股 19 个字段已由 iFinD/Datayes 切换到 Choice。
#    起点由 2003-01-02 改为 2005-01-04，股票列由 6077 改为 5855（时点并集）。
#    当前形状 5243 × 5855，覆盖 2005-01-04 ~ 2026-08-06。
#    例外：listed_days 仍来自 iFinD（5828 列，止于 2026-07-17），
#    build_investable_mask 取三者列交集，实际股票池会被它限制。
#    详见 Database/docs/【主文档】框架全景记录与Choice迁移方案.md

# -----------------------------------------------------------------------------
# 股票池过滤参数
# -----------------------------------------------------------------------------
IPO_FILTER_DAYS          = 60    # 新股过滤：上市不足 N 个交易日的股票排除
MIN_ROLLING_VALID_DAYS   = 10    # 滚动窗口内最少有效交易日（用于因子计算）

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
