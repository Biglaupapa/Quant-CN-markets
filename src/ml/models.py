# =============================================================================
# ml/models.py
# 模型库：统一 fit / predict 接口 + 验证集网格搜索
#
# 对应 GKX 的模型谱系（由简到繁），德国复现项目实现了前 7 个，本模块另加
# LightGBM（GKX 发表时尚无，但在同类任务上普遍强于 sklearn 的 GBRT，且快一个量级）：
#
#   OLS-H    全特征线性 + Huber 损失          无超参
#   OLS-3    三特征基准（size / bm / mom）    无超参，GKX 的「简单基准」
#   ENet     弹性网                           alpha × l1_ratio
#   PLS      偏最小二乘                       成分数
#   PCR      主成分回归                       主成分数
#   RF       随机森林                         深度 × 特征数
#   GBRT     梯度提升（Huber 损失）           深度 × 学习率 × 叶子样本数
#   LGBM     LightGBM（可选依赖）             同上
#
# ── 为什么调参用 Huber 损失而不是 MSE ───────────────────────────────────────
# 这是 GKX 的核心手法之一，不是可选项。个股月度收益是重尾的：本项目面板里
# 717,431 个「股票-月」观测中，有 45 个单月收益 >200%，最大 +1073%
# （688585.SH 2025-07；另有 601313.SH/601360.SH 因借壳上市在 2017-11 出现
# 同一家公司两个代码、收益逐位相同的情况）。
# 用 MSE 选超参，这几十个点会主导整个验证集的损失，选出来的参数是在拟合极端值。
# Huber 损失在 |误差| > δ 之后转为线性，把它们的影响压住。
#
# ── 标签不做缩尾 ────────────────────────────────────────────────────────────
# 德国那个复现项目把因变量也做了 1%/99% 缩尾（作者自己在注释里承认这是对
# GKX 的偏离）。本模块**不缩尾标签**，理由是 R²_oos 的分母是 Σy²，
# 缩尾会同时改变分子分母，算出来的数字无法与 GKX 的 0.3%~0.4% 对照。
# 极端值交给 Huber 损失处理——那正是它存在的意义。
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.config.compute import get_jobs

log = logging.getLogger(__name__)

# GKX 用的 Huber 阈值。sklearn 的 HuberRegressor 默认 epsilon=1.35，同源。
HUBER_DELTA = 1.35


# -----------------------------------------------------------------------------
# 损失函数
# -----------------------------------------------------------------------------

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray,
               delta: float = HUBER_DELTA) -> float:
    """Huber 损失均值。验证集选超参用这个，不用 MSE。

        |e| <= δ :  0.5 · e²
        |e|  > δ :  δ · (|e| − 0.5·δ)

    注：德国复现项目的实现写成 `δ·(|e| − 0.5δ²)`，在 δ≠1 时与标准式不同
    （量纲也不对：δ² 与 |e| 相减）。此处用标准式，保证 δ 处连续可导。
    """
    e = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    a = np.abs(e)
    return float(np.mean(np.where(a <= delta, 0.5 * e ** 2,
                                  delta * (a - 0.5 * delta))))


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(e ** 2))


LOSSES = {"huber": huber_loss, "mse": mse_loss}


# -----------------------------------------------------------------------------
# 模型规格
# -----------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """一个模型 = 构造函数 + 超参网格 + 是否需要标准化特征。

    `build(**params)` 返回一个 sklearn 风格的估计器（有 fit / predict）。
    `grid` 为空表示无超参，此时训练集与验证集**合并训练**（GKX 对 OLS 的做法）。
    """
    name: str
    build: Callable[..., Any]
    grid: dict[str, list] = field(default_factory=dict)
    features: Optional[list[str]] = None      # None = 用全部特征
    note: str = ""

    def param_list(self) -> list[dict]:
        from sklearn.model_selection import ParameterGrid
        return list(ParameterGrid(self.grid)) if self.grid else [{}]


# -----------------------------------------------------------------------------
# 各模型的构造
# -----------------------------------------------------------------------------

def _ols_huber(**kw):
    from sklearn.linear_model import HuberRegressor
    # max_iter 给足：61 个特征 + 数十万样本时默认 100 次常不收敛
    return HuberRegressor(epsilon=HUBER_DELTA, max_iter=500, alpha=1e-4, **kw)


def _enet(**kw):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(max_iter=5000, random_state=0, **kw)


def _pls(**kw):
    from sklearn.cross_decomposition import PLSRegression
    return PLSRegression(**kw)


def _pcr(n_components=10, **kw):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(PCA(n_components=n_components, random_state=0),
                         LinearRegression())


def _rf(**kw):
    from sklearn.ensemble import RandomForestRegressor
    kw.setdefault("n_estimators", 300)
    kw.setdefault("min_samples_leaf", 200)
    kw.setdefault("n_jobs", get_jobs())      # 不用 -1，见 config/compute.py
    kw.setdefault("random_state", 0)
    return RandomForestRegressor(**kw)


def _gbrt(**kw):
    """梯度提升树（sklearn 直方图实现）。

    ── 为什么不用 GradientBoostingRegressor ────────────────────────────────
    GKX 与德国复现项目用的是 sklearn 的 `GradientBoostingRegressor`（精确分裂，
    支持 loss='huber'）。在他们 214,016 行的面板上尚可，但本项目单个训练窗
    有 231,302 行 × 61 特征，实测**仅 2 组超参、单个窗口就超过 10 分钟**
    （该实现单线程、逐样本精确分裂），13 个窗口要 2 小时以上，不可行。

    改用 `HistGradientBoostingRegressor`：特征分箱 + OMP 多线程，
    与 LightGBM 同一套思路，量级上快 10~100 倍。

    ── 代价：它不支持 Huber 损失 ───────────────────────────────────────────
    可选损失只有 squared / absolute / poisson / quantile。这里取
    **absolute_error（L1）**——收益重尾，L1 对极端值的稳健性与 Huber 同源
    （Huber 本就是 L2 与 L1 的分段拼接，δ 之外那段正是 L1）。
    真正忠于 GKX 的 Huber 目标由 **LGBM** 提供（`objective="huber"`），
    两者并列，正好互为对照。

    注：它没有 n_jobs 参数，线程数由 OpenMP 决定（OMP_NUM_THREADS）。
    该变量在 `src/__init__.py` 里已按算力预算设好，见 config/compute.py。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    kw.setdefault("loss", "absolute_error")
    kw.setdefault("max_iter", 300)
    kw.setdefault("early_stopping", False)
    kw.setdefault("random_state", 0)
    # 网格里用的是 min_samples_leaf，HistGBR 的参数名是 min_samples_leaf，一致
    return HistGradientBoostingRegressor(**kw)


def _lgbm(**kw):
    import lightgbm as lgb
    kw.setdefault("objective", "huber")
    kw.setdefault("alpha", HUBER_DELTA)
    kw.setdefault("n_estimators", 500)
    kw.setdefault("subsample", 0.8)
    kw.setdefault("subsample_freq", 1)
    kw.setdefault("colsample_bytree", 0.7)
    kw.setdefault("n_jobs", get_jobs())      # 同上；LightGBM 内部即 num_threads
    kw.setdefault("random_state", 0)
    kw.setdefault("verbose", -1)
    return lgb.LGBMRegressor(**kw)


def has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


# -----------------------------------------------------------------------------
# 滚动符号等权 —— 本项目最关键的对照组
# -----------------------------------------------------------------------------

class SignEqualWeight:
    """方向对齐后**等权平均**，方向只由训练数据决定。

    ── 为什么必须有这个模型 ────────────────────────────────────────────────
    没有它，就无法区分 ML 的增量到底来自哪里。实测已知：
      · 在合成池那 10 个特征上，学习**打不过**等权（夏普 1.0 vs 1.19）
      · 换到 61 个特征，ML 夏普跳到 2.0~2.3
    这两件事同时成立时，有两种互斥的解释：
      (a) 宽度本身就够——等权用 61 个特征也能到 2.0，模型没贡献
      (b) 学习需要足够宽度才发挥——等权只能到 1.4，模型确实有贡献
    本模型就是 (a) 的实现：**同样 61 个特征、同样滚动窗口，但权重恒等权**。
    它与 ML 的差值，才是「学习」的净贡献。

    ── 为什么方向要滚动决定 ────────────────────────────────────────────────
    既有 `main.py` 的 `FACTOR_DIRECTIONS` 是人工按**全样本** IC 填的，
    直接拿来用就是前视偏差。这里改为：每个窗口只用**训练段**的 IC 符号，
    与 ML 模型看到的信息完全一致，对照才公平。

    ── 为什么还要拟合 a、b ─────────────────────────────────────────────────
    等权得分是 [-1,1] 区间的无量纲数，不是收益率的估计。直接拿去算 R²_oos
    会因为尺度完全不对而得到巨大负值，没有意义。故在训练段上再拟合一个
    一元线性映射 ŷ = a + b·score，把它放到收益率量纲上。
    这只有 2 个参数，**不改变任何排序**（b>0 时 IC、分组结果与原始得分完全相同），
    因此它仍然是「等权」，只是变成了可比的收益预测。
    """

    def __init__(self, min_abs_ic: float = 0.0):
        self.min_abs_ic = min_abs_ic
        self.signs_: Optional[np.ndarray] = None
        self.a_: float = 0.0
        self.b_: float = 1.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # 各特征与标签的相关系数符号（Pearson 即可——特征已是秩变换后的均匀分布）
        Xc = X - X.mean(axis=0, keepdims=True)
        yc = y - y.mean()
        denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = np.where(denom > 0, (Xc * yc[:, None]).sum(axis=0) / denom, 0.0)

        s = np.sign(ic)
        s[np.abs(ic) < self.min_abs_ic] = 0.0      # 太弱的特征直接不投票
        self.signs_ = s

        score = self._score(X)
        # 一元线性映射到收益量纲
        sc = score - score.mean()
        var = float((sc ** 2).sum())
        self.b_ = float((sc * yc).sum() / var) if var > 0 else 0.0
        self.a_ = float(y.mean() - self.b_ * score.mean())
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        n = np.count_nonzero(self.signs_)
        if n == 0:
            return np.zeros(len(X))
        return (X * self.signs_).sum(axis=1) / n

    def predict(self, X):
        return self.a_ + self.b_ * self._score(np.asarray(X, dtype=np.float64))


# -----------------------------------------------------------------------------
# 默认模型集
# -----------------------------------------------------------------------------
#
# ⚠️ 超参网格必须按**本项目的样本量**标定，不能照抄德国那份。
# 那个项目是 214,016 个观测，其 GBRT 网格用 min_samples_split ∈
# {5000, 8000, 10000}；本项目面板 716,052 行（3.3 倍），照抄会严重欠拟合。
# 这里按样本量等比放大，并改用 min_samples_leaf（更直接控制叶子粒度）。

def default_specs(features: list[str],
                  include_lgbm: bool = True,
                  fast: bool = False) -> list[ModelSpec]:
    """默认模型集。`fast=True` 时缩小网格，用于跑通验证。"""

    # OLS-3：GKX 的「简单基准」——只用 size / bm / 动量三个特征。
    # 本框架里动量整体失效，取 IC 最强的 reversal_20 代替 mom（并注明）。
    three = [c for c in ("size", "bm", "reversal_20") if c in features]

    specs = [
        # ★ 对照组放第一位：它回答「学习到底有没有贡献」
        ModelSpec("EW-Sign", lambda **kw: SignEqualWeight(**kw), {}, None,
                  "滚动符号等权（对照组）——同样的特征与窗口，但权重恒等权。"
                  "与其他模型的差值 = 学习的净贡献"),
        ModelSpec("OLS-H", _ols_huber, {}, None,
                  "全特征线性 + Huber 损失；无超参，train+val 合并训练"),
        ModelSpec("OLS-3", _ols_huber, {}, three or None,
                  f"简单基准，仅 {three}（GKX 用 size/bm/mom，"
                  f"A股动量失效故以 reversal_20 代替）"),
        ModelSpec("ENet", _enet,
                  {"alpha": [1e-4, 1e-3] if fast else [1e-5, 1e-4, 1e-3, 1e-2],
                   "l1_ratio": [0.5] if fast else [0.1, 0.5, 0.9]}),
        ModelSpec("PLS", _pls,
                  {"n_components": [3, 6] if fast else [1, 2, 3, 5, 8, 12, 20]}),
        ModelSpec("PCR", _pcr,
                  {"n_components": [5, 15] if fast else [2, 5, 10, 15, 20, 30]}),
        ModelSpec("RF", _rf,
                  {"max_depth": [4, 6] if fast else [2, 4, 6, 8],
                   "max_features": [10] if fast else [5, 10, 20, 40]}),
        ModelSpec("GBRT", _gbrt,
                  {"max_depth": [2, 4] if fast else [1, 2, 3, 4, 6],
                   "learning_rate": [0.05] if fast else [0.01, 0.05, 0.1],
                   "min_samples_leaf": [500] if fast else [200, 500, 1000]}),
    ]
    if include_lgbm and has_lightgbm():
        specs.append(ModelSpec("LGBM", _lgbm,
                               {"max_depth": [4, 6] if fast else [3, 4, 6, 8],
                                "learning_rate": [0.05] if fast else [0.01, 0.05, 0.1],
                                "min_child_samples": [500] if fast else [200, 500, 1000]},
                               None, "GKX 发表时尚无；同类任务上通常强于 sklearn GBRT"))
    elif include_lgbm:
        log.info("[models] 未安装 lightgbm，跳过 LGBM（pip install lightgbm 可启用）")
    return specs


# -----------------------------------------------------------------------------
# 验证集调参 + 拟合
# -----------------------------------------------------------------------------

def fit_with_validation(spec: ModelSpec,
                        X_tr: np.ndarray, y_tr: np.ndarray,
                        X_va: np.ndarray, y_va: np.ndarray,
                        loss: str = "huber") -> tuple[Any, dict, float]:
    """在验证集上按 `loss` 选超参，返回 (已拟合模型, 最优参数, 验证损失)。

    无超参的模型（OLS）走 GKX 的做法：**train + val 合并训练**——
    既然没有要调的东西，验证集就没必要空着，多 24 个月的数据更有价值。

    有超参的模型：先在 train 上逐组拟合、在 val 上评分选出最优，
    然后**用最优参数在 train+val 上重新拟合一次**再去预测测试集。
    重拟合是刻意的：调参已经用掉了 val 的信息，不重拟合等于白扔 24 个月。
    """
    fn = LOSSES[loss]
    params = spec.param_list()

    if len(params) == 1:
        model = spec.build(**params[0])
        Xf = np.vstack([X_tr, X_va])
        yf = np.concatenate([y_tr, y_va])
        model.fit(Xf, yf)
        return model, params[0], fn(y_va, np.asarray(model.predict(X_va)).ravel())

    best, best_p, best_l = None, None, np.inf
    for p in params:
        try:
            m = spec.build(**p)
            m.fit(X_tr, y_tr)
            l = fn(y_va, np.asarray(m.predict(X_va)).ravel())
        except Exception as e:
            log.debug("[models] %s %s 拟合失败：%s", spec.name, p, e)
            continue
        if l < best_l:
            best, best_p, best_l = m, p, l

    if best is None:
        raise RuntimeError(f"[models] {spec.name} 所有超参组合都失败了")

    final = spec.build(**best_p)
    final.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    return final, best_p, best_l


def predict(model, X: np.ndarray) -> np.ndarray:
    """统一预测出口。PLS 返回 (n,1)，这里一律拉平成 (n,)。"""
    return np.asarray(model.predict(X)).ravel()
