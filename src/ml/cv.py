# =============================================================================
# ml/cv.py
# 滚动窗口切分（训练 / 验证 / 测试严格时序隔离）
#
# 对应 GKX 与德国复现项目里的 `TimeBasedCV`，但改成以**月末 DatetimeIndex**
# 为单位，与本框架的月频面板对齐（原实现按 dateutil.relativedelta 逐日推）。
#
# ── 这是整个 ML 管线最关键的一块 ────────────────────────────────────────────
# 它决定了「样本外」这三个字是否成立。任何一处泄漏——验证集参与训练、
# 测试集参与调参、窗口跨越标签的未来——都会让 R²_oos 变成一个漂亮但无效的数字。
#
# 时序结构（默认 60 / 24 / 12，与 GKX 一致）：
#
#   |<--- train 60m --->|<- val 24m ->|<- test 12m ->|
#                       ^调参用          ^只做预测，绝不回看
#   窗口每次前滚 step 个月（默认 12 = 每年重估一次）
#
# ── 一个必须理解的细节：标签的一个月错位 ────────────────────────────────────
# 面板里 T 月那一行的标签 y 是 **T+1 月**的收益（`ret.shift(-1)`）。
# 因此训练集最后一个月 T_end 的标签用到了 T_end+1 月的信息，
# 而验证集从 T_end+1 开始——**训练标签与验证特征落在同一个月**。
# 这不是泄漏（用的是不同变量：一个是 T_end+1 的收益、一个是 T_end+1 的特征），
# 但若要绝对保守，可设 `gap=1` 在各段之间空出一个月。默认 gap=0，与 GKX 一致。
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd


@dataclass
class Split:
    """一个滚动窗口。三段互不相交，且严格按时间先后排列。"""
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def __repr__(self) -> str:
        f = lambda ix: f"{ix[0]:%Y-%m}~{ix[-1]:%Y-%m}" if len(ix) else "空"
        return (f"Split(train={f(self.train)}[{len(self.train)}m] "
                f"val={f(self.val)}[{len(self.val)}m] "
                f"test={f(self.test)}[{len(self.test)}m])")


class RollingWindowCV:
    """滚动窗口切分器。

    Parameters
    ----------
    train_months, val_months, test_months : 三段的长度（月）
    step_months : 窗口每次前滚多少个月。默认 = test_months，
        保证各窗口的测试集**首尾相接、不重不漏**，
        拼起来正好是一段连续的样本外区间。
    gap : 各段之间空出的月数（见模块文档的「标签错位」说明）
    expanding : True = 训练集起点固定不动（扩展窗口）——**GKX 2020 附录 D 的做法**
        （"recursively increasing the training sample"，验证集定长前滚）；
        False = 训练集长度固定（滚动窗口）。构造器默认 False 仅为向后兼容，
        项目默认值见 run.ML_CONFIG（2026-10-03 起为 True）
    """

    def __init__(self,
                 train_months: int = 60,
                 val_months: int = 24,
                 test_months: int = 12,
                 step_months: Optional[int] = None,
                 gap: int = 0,
                 expanding: bool = False):
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months if step_months is not None else test_months
        self.gap = gap
        self.expanding = expanding

    @property
    def warmup(self) -> int:
        """第一个测试月之前需要消耗掉的月数。"""
        return self.train_months + self.val_months + 2 * self.gap

    def n_splits(self, dates: pd.DatetimeIndex) -> int:
        """窗口数。**含末尾的残窗**——最后一个窗口的测试段可能不足
        `test_months` 个月（数据到头了），但它仍是一个有效窗口，
        丢掉它等于白白扔掉最近的样本外月份。"""
        avail = len(dates) - self.warmup
        if avail <= 0:
            return 0
        return int(np.ceil(avail / self.step_months))

    def split(self, dates: pd.DatetimeIndex) -> Iterator[Split]:
        """按时间顺序产出各个窗口。

        `dates` 必须是**去重且升序**的月末索引（面板的 date level）。
        """
        dates = pd.DatetimeIndex(pd.unique(dates)).sort_values()
        n = len(dates)
        if self.n_splits(dates) == 0:
            raise ValueError(
                f"[cv] 数据不足：共 {n} 个月，而一个窗口至少需要 "
                f"{self.warmup + self.test_months} 个月"
                f"（train{self.train_months}+val{self.val_months}"
                f"+test{self.test_months}+gap{2*self.gap}）")

        start = 0
        while True:
            tr_lo = 0 if self.expanding else start
            tr_hi = start + self.train_months
            va_lo = tr_hi + self.gap
            va_hi = va_lo + self.val_months
            te_lo = va_hi + self.gap
            te_hi = te_lo + self.test_months
            if te_lo >= n:
                break
            yield Split(train=dates[tr_lo:tr_hi],
                        val=dates[va_lo:va_hi],
                        test=dates[te_lo:min(te_hi, n)])
            start += self.step_months
            if start + self.warmup >= n:
                break

    def describe(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """把切分方案列成表，跑之前先看一眼，确认没有意外。"""
        rows = []
        for i, s in enumerate(self.split(dates), 1):
            rows.append({
                "窗口": i,
                "训练": f"{s.train[0]:%Y-%m} ~ {s.train[-1]:%Y-%m}",
                "验证": f"{s.val[0]:%Y-%m} ~ {s.val[-1]:%Y-%m}",
                "测试": f"{s.test[0]:%Y-%m} ~ {s.test[-1]:%Y-%m}",
                "训练月": len(s.train), "验证月": len(s.val), "测试月": len(s.test),
            })
        return pd.DataFrame(rows)


def assert_no_leakage(split: Split) -> None:
    """断言三段严格递增且互不相交。管线里每个窗口都跑一次，成本可忽略。"""
    if len(split.train) and len(split.val):
        assert split.train[-1] < split.val[0], \
            f"[cv] 训练集末({split.train[-1]:%Y-%m}) >= 验证集首({split.val[0]:%Y-%m})"
    if len(split.val) and len(split.test):
        assert split.val[-1] < split.test[0], \
            f"[cv] 验证集末({split.val[-1]:%Y-%m}) >= 测试集首({split.test[0]:%Y-%m})"
    for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
        ia, ib = getattr(split, a), getattr(split, b)
        overlap = ia.intersection(ib)
        assert len(overlap) == 0, f"[cv] {a} 与 {b} 有 {len(overlap)} 个月重叠"
