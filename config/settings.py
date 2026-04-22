# =============================================================================
# config/settings.py
# 全局路径与参数配置
# =============================================================================

from pathlib import Path

# -----------------------------------------------------------------------------
# 项目根目录
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

# -----------------------------------------------------------------------------
# 数据路径
# -----------------------------------------------------------------------------

# 历史存档数据（原始 CSV，2014-2020）
ARCHIVE_DATA_DIR = PROJECT_ROOT / "_archive" / "raw_data"

# Database：iFinD/Wind 实时更新数据（2021-至今）
DATABASE_DIR = Path("/Users/louisliu/Mirror/MyProjects/Database/data/stock/A")

# 行业因子载荷（HDF5，用于行业中性化，来自原始框架）
INDUSTRY_H5_PATH = ARCHIVE_DATA_DIR / "FactorLoading_Industry_arch.h5"

# 因子计算结果输出目录
FACTOR_OUTPUT_DIR = PROJECT_ROOT / "factors" / "output"
FACTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AF-pricing 因子方法参考路径（不复制代码，仅作来源注释）
AF_PRICING_CODING_DIR = Path("/Users/louisliu/Mirror/MyProjects/AF-pricing/Coding")

# -----------------------------------------------------------------------------
# 存档数据字段映射
# 原始 CSV 格式：index=股票代码, columns=日期（转置格式）
# -----------------------------------------------------------------------------
ARCHIVE_FIELD_MAP = {
    "close_adj":      "后复权收盘价.csv",      # 后复权收盘价（用于收益率计算）
    "open_adj":       "后复权开盘价.csv",       # 后复权开盘价（用于月初基准）
    "close":          "不复权收盘价.csv",       # 不复权收盘价（用于市值计算）
    "turn":           "换手率.csv",             # 换手率
    "pb":             "PB.csv",                 # 市净率
    "float_shares":   "流通股本.csv",           # 流通股本（股数）
    "is_st":          "是否ST股.csv",           # ST 标记（1=ST，0=正常）
    "trade_status":   "交易状态.csv",           # 交易状态（"交易"=正常）
    "listing_days":   "上市交易日数.csv",       # 上市至今交易日数
    "net_profit":     "归属母公司净利润.csv",   # 归母净利润（季度）
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
    # 基本面（快照）
    "pe_ttm":          "pe_ttm.csv",
    "pb":              "pb.csv",
    "dividend_ratio":  "dividend_ratio.csv",
    # 股票池过滤（原为存档专属，现已补入 Database）
    # Database 中字段名与存档不同，此处统一映射
    "is_st":           "st.csv",               # 1=ST，0=正常
    "trade_status":    "status.csv",           # 1=正常交易，0=停牌（与存档字符串格式不同）
    "listing_days":    "listed_days.csv",      # 上市至今交易日数
    "float_shares":    "free_float_shares.csv", # 流通股本（股）
}

# -----------------------------------------------------------------------------
# 时间范围
# -----------------------------------------------------------------------------
HIST_START = "2014-01-01"   # 历史 CSV 数据起始（用于存档 adjusted 价格）
HIST_END   = "2021-03-31"   # 历史 CSV 数据截止（存档实际到 2021-03-31）
DB_START   = "2003-01-01"   # Database 数据起始（实际从 2003-01-02 开始，含完整历史）
# 注意：Database 含 OHLCV+换手率+成交额 的完整历史（2003-至今），
#       但 PE/PB/股息率 仅有约 6 天快照（2025-06），暂不可用于回测。

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
