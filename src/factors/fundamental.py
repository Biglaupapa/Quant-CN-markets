# =============================================================================
# factors/fundamental.py
# 基本面因子库
#
# 包含所有财报/估值衍生的基本面信号：
#
#   pb               市净率（月末截面，无中性化）
#   bm               账面市值比 log(1/PB)（月末截面，无中性化，正向因子）
#   pe_ttm           市盈率 TTM（月末截面，无中性化）
#   pe1              市盈率（月末截面，无中性化）
#                    ⚠️ 名字是历史遗留，2026-08-17 起实为 Choice PE（静态口径），
#                       不再是原 Datayes 的单季年化动态 PE。详见 calc_pe1 docstring
#   dividend_yield   股息率（月末截面，无中性化）
#   size             市值因子 log(总市值)（月末截面，无中性化）
#                    → 数据源：market_value
#   size2            流通市值因子 log(流通市值)（月末截面，无中性化）
#                    → 数据源：neg_market_value
#   net_profit_yoy   净利润同比增速（季度，滞后3个季度，流通市值+行业双重中性化）
#
#   ── 2026-08-17 新增（均来自 Choice）────────────────────────────────────────
#   ps_ttm           市销率 TTM（月末截面，无中性化）覆盖 100%
#   ev_ebitda        企业倍数 EV2/EBITDA（月末截面，无中性化）覆盖 98%
#   est_pe_ftm       预测市盈率（未来12月，月末截面，无中性化）⚠️ 覆盖仅约 51%
#   est_peg          预测 PEG（月末截面，无中性化）⚠️ 覆盖仅约 51%
#   ev2_neutral      log(企业价值剔除货币资金)（月末截面，**流通市值中性化**）
#
# 中性化说明：
#   net_profit_yoy 做流通市值+行业双重中性化
#   ev2_neutral    做流通市值中性化（EV2 是水平量，不中性化即为市值代理）
#   其余因子均不做中性化
#
# 数据来源标注：
#   [arch]   来自 _archive/raw_data/ 历史 CSV（2014-2020），现仅 net_profit 依赖
#   [db]     来自 Database/data/stock/A/（2005-至今，已 100% Choice）
#   [orig]   移植自原 因子框架.py
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional
import warnings

from src.data.loader import load_data, to_monthly
from src.data.universe import apply_universe, build_investable_mask
from src.factors.base import preprocess
from src.config.settings import (
    NP_YOY_LAG_QUARTERS,
    MIN_ROLLING_VALID_DAYS,
    INDUSTRY_PATH,
)


# -----------------------------------------------------------------------------
# 辅助：构建 log(市值) 用于中性化与 Size 因子
# -----------------------------------------------------------------------------

def _get_log_mktcap(
    start: Optional[str],
    end: Optional[str],
    mask: pd.DataFrame,
    use_field: str = "neg_market_value",
) -> Optional[pd.DataFrame]:
    """
    计算月度 log(市值)，直接使用 Datayes 提供的市值字段。

    Args:
        use_field: "market_value"    （总市值，用于 Size 因子）
                   "neg_market_value"（流通市值，默认，用于 Size2 及市值中性化）

    数据覆盖：Datayes market_value / neg_market_value 均从 2004-01-02 起可用。
    """
    data = load_data([use_field], start=start, end=end)
    mv = data.get(use_field)

    if mv is None or mv.empty:
        warnings.warn(f"[fundamental] 无法获取市值数据：{use_field} 为空。")
        return None

    mv_m = to_monthly(apply_universe(mv, mask), method="last").replace(0, np.nan)
    if not mv_m.notna().any().any():
        warnings.warn(f"[fundamental] {use_field} 月度数据全为空。")
        return None

    return np.log(mv_m)


# -----------------------------------------------------------------------------
# 1. 市净率因子（PB）
#    来源：[arch] PB.csv / [db] pb.csv
# -----------------------------------------------------------------------------

def calc_pb(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市净率因子（PB）：月末截面值，越小代表越"价值"。
    无中性化。

    数据：pb [arch: PB.csv] / [db: pb.csv]
    """
    data = load_data(["pb"], start=start, end=end)
    pb = data.get("pb")

    if pb is None:
        warnings.warn("[fundamental] calc_pb: 缺少 pb 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    pb_m = to_monthly(apply_universe(pb, mask), method="last")

    return preprocess(pb_m)


# -----------------------------------------------------------------------------
# 2. 账面市值比因子（B/M）
#    来源：[arch] PB.csv / [db] pb.csv（取倒数后取对数）
# -----------------------------------------------------------------------------

def calc_bm(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    账面市值比因子（B/M = 1/PB）：log(1/PB) = -log(PB)。
    越大代表估值越低（价值股），正向因子（高 B/M → 高预期收益）。
    参考 Fama-French HML 构建逻辑，无中性化。

    数据：pb [arch: PB.csv] / [db: pb.csv]
    """
    data = load_data(["pb"], start=start, end=end)
    pb = data.get("pb")

    if pb is None:
        warnings.warn("[fundamental] calc_bm: 缺少 pb 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    pb_m = to_monthly(apply_universe(pb, mask), method="last")

    # PB <= 0 无意义，置 NaN
    pb_m[pb_m <= 0] = np.nan

    # B/M = log(1/PB) = -log(PB)
    bm_m = -np.log(pb_m)

    return preprocess(bm_m)


# -----------------------------------------------------------------------------
# 3. 市盈率因子（PE TTM）
#    来源：[db] pe_ttm.csv
# -----------------------------------------------------------------------------

def calc_pe_ttm(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市盈率 TTM 因子：月末截面值，越小代表越"价值"。
    负值（亏损公司）置为 NaN。无中性化。

    数据：pe_ttm [db: pe_ttm.csv]（Datayes，2004+）
    """
    data = load_data(["pe_ttm"], start=start, end=end)
    pe = data.get("pe_ttm")

    if pe is None:
        warnings.warn("[fundamental] calc_pe_ttm: 缺少 pe_ttm 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    pe_m = to_monthly(apply_universe(pe, mask), method="last")
    pe_m[pe_m <= 0] = np.nan

    return preprocess(pe_m)


# -----------------------------------------------------------------------------
# 4. 动态市盈率因子（PE1）
#    来源：[db] pe1.csv（Datayes getMktEqud.PE1）
# -----------------------------------------------------------------------------

def calc_pe1(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市盈率因子：月末截面值，越小代表估值越低。负值（亏损公司）置为 NaN。无中性化。

    ⚠️ 口径已于 2026-08-17 变更，因子名是历史遗留：
      旧（Datayes pe1）= 市值 / (最新单季净利 × 4)，单季年化动态 PE
      新（Choice PE） = 市值 / 最近年报净利，静态 PE

    换源原因：Datayes 停更于 2026-07-17，而 Choice 的 26 个字段里没有净利润，
    无法重建单季年化口径（与 PE 秩相关仅 0.20~0.43）。因此这不是无损换源，
    该因子的历史值已整体改变，不能与 2026-08-17 之前的回测数字直接比较。

    数据：pe1 [db: pe.csv]（Choice PE，2005+）
    """
    data = load_data(["pe1"], start=start, end=end)
    pe1 = data.get("pe1")

    if pe1 is None:
        warnings.warn("[fundamental] calc_pe1: 缺少 pe1 数据")
        return pd.DataFrame()

    mask  = build_investable_mask(start=start, end=end, freq="D")
    pe1_m = to_monthly(apply_universe(pe1, mask), method="last")
    pe1_m[pe1_m <= 0] = np.nan

    return preprocess(pe1_m)


# -----------------------------------------------------------------------------
# 5. 股息率因子（Dividend Yield）
#    来源：[db] dividend_ratio.csv
# -----------------------------------------------------------------------------

def calc_dividend_yield(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    股息率 TTM 因子：月末截面值，越高代表股息越丰厚。
    无中性化。

    数据：dividend_ratio [db: dividend_ratio.csv]（Datayes，2004+）
    """
    data = load_data(["dividend_ratio"], start=start, end=end)
    div = data.get("dividend_ratio")

    if div is None:
        warnings.warn("[fundamental] calc_dividend_yield: 缺少 dividend_ratio 数据")
        return pd.DataFrame()

    mask  = build_investable_mask(start=start, end=end, freq="D")
    div_m = to_monthly(apply_universe(div, mask), method="last")

    return preprocess(div_m)


# -----------------------------------------------------------------------------
# 6. 市值因子（SIZE）— 总市值
#    数据：market_value（Datayes getMktDivYield，2004+）
# -----------------------------------------------------------------------------

def calc_size(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Size 因子：log(总市值)，月末取值。
    不做中性化（市值因子本身是中性化的控制变量，不宜自我中性化）。

    数据：market_value（Datayes，2004+）
    """
    mask = build_investable_mask(start=start, end=end, freq="D")
    log_mktcap = _get_log_mktcap(start, end, mask, use_field="market_value")

    if log_mktcap is None:
        warnings.warn("[fundamental] calc_size: 无法计算 log(总市值)")
        return pd.DataFrame()

    return preprocess(log_mktcap, neutralize=None)


# -----------------------------------------------------------------------------
# 7. 流通市值因子（SIZE2）— 流通市值
#    数据：neg_market_value（Datayes getMktEqud，2004+）
# -----------------------------------------------------------------------------

def calc_size2(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Size2 因子：log(流通市值)，月末取值。
    与 Size（总市值）的差异体现在限售股比例较高的个股上。
    不做中性化。

    数据：neg_market_value（Datayes，2004+）
    """
    mask = build_investable_mask(start=start, end=end, freq="D")
    log_mktcap = _get_log_mktcap(start, end, mask, use_field="neg_market_value")

    if log_mktcap is None:
        warnings.warn("[fundamental] calc_size2: 无法计算 log(流通市值)")
        return pd.DataFrame()

    return preprocess(log_mktcap, neutralize=None)


# -----------------------------------------------------------------------------
# 8. 净利润同比增速（NetProfit YoY Growth）
#    来源：[arch] 归属母公司净利润.csv + [orig] 因子框架.py
# -----------------------------------------------------------------------------

def calc_net_profit_yoy(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    净利润同比增速：当季净利润 / 上年同季净利润 - 1。

    处理细节（延续 因子框架.py 做法）：
    - 数据为季度频率，滞后 NP_YOY_LAG_QUARTERS 个季度（财报披露延迟，默认3季度）
    - 即：在 t 月使用 t-3 季度的财务数据，确保无前瞻偏差
    - 去极值后做流通市值+行业双重中性化

    注：目前该数据仅存在于存档 CSV（归属母公司净利润.csv），
        Database 尚未覆盖净利润季度数据，后续补充后可扩展。

    数据：net_profit [arch: 归属母公司净利润.csv]（季度）
    """
    data = load_data(["net_profit"], start=start, end=end)
    net_profit = data.get("net_profit")

    if net_profit is None:
        warnings.warn("[fundamental] calc_net_profit_yoy: 缺少 net_profit 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")

    # 净利润为季度数据，对齐后计算同比增速
    np_monthly = net_profit  # 保持原始季度频率

    # 应用股票池掩码（月末掩码对齐）
    mask_m = mask.resample("ME").last().fillna(False)

    # 同比增速：当期 / 上年同期 - 1（即 shift(4) 对季度数据）
    yoy = np_monthly / np_monthly.shift(4) - 1

    # 滞后 NP_YOY_LAG_QUARTERS 个季度（财报披露延迟）
    yoy = yoy.shift(NP_YOY_LAG_QUARTERS)

    # 去除极端值（同比增速可能因基期接近零而出现极大值）
    yoy = yoy.replace([np.inf, -np.inf], np.nan)

    # 月度对齐（季度末填充到月度）
    yoy = yoy.resample("ME").last().ffill(limit=2)

    # 应用股票池掩码
    common_idx  = yoy.index.intersection(mask_m.index)
    common_cols = yoy.columns.intersection(mask_m.columns)
    yoy = yoy.reindex(index=common_idx, columns=common_cols)
    mask_aligned = mask_m.reindex(index=common_idx, columns=common_cols)
    yoy[~mask_aligned] = np.nan

    # 流通市值+行业双重中性化
    log_mktcap = _get_log_mktcap(start, end, mask, use_field="neg_market_value")

    return preprocess(
        yoy,
        neutralize="size+industry",
        log_mktcap=log_mktcap,
        industry_path=INDUSTRY_PATH,
    )


# -----------------------------------------------------------------------------
# 9. 市销率因子（PS TTM）
#    来源：[db] ps_ttm.csv（Choice PSTTM）
# -----------------------------------------------------------------------------

def calc_ps_ttm(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    市销率 TTM 因子：月末截面值，越小代表估值越低。

    相比 PE，PS 的分母是营收而非利润，亏损公司同样有值——2026-08-14 截面
    非空 100%、负值 0.0%，是这批估值字段里最干净的一个。

    负值置为 NaN（理论上不存在，防御性处理）。无中性化。

    数据：ps_ttm [db: ps_ttm.csv]（Choice PSTTM，2005+）
    """
    data = load_data(["ps_ttm"], start=start, end=end)
    ps = data.get("ps_ttm")

    if ps is None:
        warnings.warn("[fundamental] calc_ps_ttm: 缺少 ps_ttm 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    ps_m = to_monthly(apply_universe(ps, mask), method="last")
    ps_m[ps_m <= 0] = np.nan

    return preprocess(ps_m)


# -----------------------------------------------------------------------------
# 10. 企业倍数因子（EV2 / EBITDA）
#     来源：[db] ev_ebitda.csv（Choice EVTOEBITDA）
# -----------------------------------------------------------------------------

def calc_ev_ebitda(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    企业倍数因子 EV/EBITDA：月末截面值，越小代表估值越低。

    相比 PE，EV/EBITDA 剔除了资本结构与折旧摊销的影响，跨行业可比性更好。
    2026-08-14 截面非空 98.1%，其中 16.9% 为负（EBITDA 为负的亏损公司），
    与 PE 一样置为 NaN。无中性化。

    数据：ev_ebitda [db: ev_ebitda.csv]（Choice EVTOEBITDA，2005+）
    """
    data = load_data(["ev_ebitda"], start=start, end=end)
    ev = data.get("ev_ebitda")

    if ev is None:
        warnings.warn("[fundamental] calc_ev_ebitda: 缺少 ev_ebitda 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    ev_m = to_monthly(apply_universe(ev, mask), method="last")
    ev_m[ev_m <= 0] = np.nan

    return preprocess(ev_m)


# -----------------------------------------------------------------------------
# 11. 预测市盈率因子（PE, 未来12月）
#     来源：[db] est_pe_ftm.csv（Choice ESTPEFTM）
# -----------------------------------------------------------------------------

def calc_est_pe_ftm(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    预测市盈率因子（未来12个月）：月末截面值，越小代表估值越低。

    与 pe_ttm 的区别在于分母是分析师一致预期盈利而非已实现盈利，含前瞻信息。

    ⚠️ 覆盖率仅约 51%——只有被分析师跟踪的标的才有值，天然偏向大盘股。
    该因子实际只对半个市场发声，且这半个市场不是随机抽样，解读时须注意选择偏差。

    负值置为 NaN。无中性化。

    数据：est_pe_ftm [db: est_pe_ftm.csv]（Choice ESTPEFTM，2005+）
    """
    data = load_data(["est_pe_ftm"], start=start, end=end)
    pe = data.get("est_pe_ftm")

    if pe is None:
        warnings.warn("[fundamental] calc_est_pe_ftm: 缺少 est_pe_ftm 数据")
        return pd.DataFrame()

    mask = build_investable_mask(start=start, end=end, freq="D")
    pe_m = to_monthly(apply_universe(pe, mask), method="last")
    pe_m[pe_m <= 0] = np.nan

    return preprocess(pe_m)


# -----------------------------------------------------------------------------
# 12. 预测 PEG 因子
#     来源：[db] est_peg.csv（Choice ESTPEG）
# -----------------------------------------------------------------------------

def calc_est_peg(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    预测 PEG 因子：预测 PE / 预测盈利增速，越小代表「成长性相对估值」越便宜。

    ⚠️ 两重注意：
      1. 覆盖率与 est_pe_ftm 相同（约 51%），同样偏向大盘股
      2. PEG 是「比率的比率」，分母增速接近 0 时数值会爆炸，噪声天然大于 PE

    负值置为 NaN（增速为负时 PEG 无经济含义）。无中性化。

    数据：est_peg [db: est_peg.csv]（Choice ESTPEG，2005+）
    """
    data = load_data(["est_peg"], start=start, end=end)
    peg = data.get("est_peg")

    if peg is None:
        warnings.warn("[fundamental] calc_est_peg: 缺少 est_peg 数据")
        return pd.DataFrame()

    mask  = build_investable_mask(start=start, end=end, freq="D")
    peg_m = to_monthly(apply_universe(peg, mask), method="last")
    peg_m[peg_m <= 0] = np.nan

    return preprocess(peg_m)


# -----------------------------------------------------------------------------
# 13. 企业价值因子（log EV2，市值中性化）
#     来源：[db] ev2.csv（Choice EV2）
# -----------------------------------------------------------------------------

def calc_ev2_neutral(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    企业价值（剔除货币资金）因子：log(EV2) 后做流通市值中性化。

    ⚠️ 为什么必须中性化：EV2 是以「元」为单位的水平量（2026-08-14 截面中位数
    约 62 亿），不是比率。原样使用本质上就是一个市值代理，会与 size 因子高度共线，
    跑出来的信号只是规模溢价的复制品。取对数后对 log(流通市值) 回归取残差，
    剥离规模成分，剩下的才是「同等规模下企业价值偏高/偏低」这一独立信息。

    非正值置为 NaN（取对数要求），中性化方式：size。

    数据：ev2 [db: ev2.csv]（Choice EV2，2005+）+ neg_market_value
    """
    data = load_data(["ev2"], start=start, end=end)
    ev2 = data.get("ev2")

    if ev2 is None:
        warnings.warn("[fundamental] calc_ev2_neutral: 缺少 ev2 数据")
        return pd.DataFrame()

    mask   = build_investable_mask(start=start, end=end, freq="D")
    ev2_m  = to_monthly(apply_universe(ev2, mask), method="last")
    ev2_m[ev2_m <= 0] = np.nan
    log_ev2 = np.log(ev2_m)

    log_mktcap = _get_log_mktcap(start, end, mask, use_field="neg_market_value")

    return preprocess(
        log_ev2,
        neutralize="size",
        log_mktcap=log_mktcap,
    )
