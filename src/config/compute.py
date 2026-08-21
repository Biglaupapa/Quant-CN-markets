# =============================================================================
# config/compute.py
# 算力预算 —— 控制整个框架用多少 CPU 线程
#
# ── 为什么需要这个模块 ──────────────────────────────────────────────────────
# 原先多处写死「用满所有核心」：
#   · models.py 的 RF / LGBM：n_jobs=-1
#   · HistGradientBoostingRegressor：OpenMP 默认开满逻辑核
#   · numpy / scipy 背后的 BLAS（Accelerate / OpenBLAS）：同样默认开满
# 在 MacBook 上连续跑几十分钟滚动训练，风扇全开、机身烫手，还会因为热
# 降频反而拖慢整体（P 核降频后单窗耗时不降反升）。
#
# 本模块把线程数收敛成**一个可配置的预算**，默认取逻辑核的一半，
# 留出余量给系统与其他程序，机器明显更凉。
#
# ── 三个层次，缺一不可 ──────────────────────────────────────────────────────
#   1. 环境变量（OMP_NUM_THREADS 等）：必须在 numpy / sklearn **导入之前**
#      设置才生效，故由 `src/__init__.py` 在包导入的第一时间调用。
#   2. threadpoolctl：对**已经加载**的 BLAS / OpenMP 动态库运行时改限额，
#      兜住第 1 步来不及的情况（例如在 notebook 里已经 import 过 numpy）。
#   3. get_jobs()：给 sklearn / LightGBM 的 n_jobs 参数用。
#
# ── 用法 ────────────────────────────────────────────────────────────────────
#   python -m src.ml.run --jobs 4        # 本次只用 4 线程
#   QUANT_JOBS=4  python -m src.main     # 环境变量，对所有入口生效
#   QUANT_JOBS=all python -m src.ml.run  # 恢复「用满」的旧行为
#
# 注：限线程降的是**峰值功耗**。长时间连续训练仍会积热，需要更凉时配合
# `--cooldown`（窗口之间停几秒散热，见 ml/run.py）。
# =============================================================================

from __future__ import annotations

import os
import subprocess

# 这些环境变量必须在 numpy / sklearn 导入前设置才生效
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",          # OpenMP：HistGradientBoosting、LightGBM
    "OPENBLAS_NUM_THREADS",     # OpenBLAS
    "MKL_NUM_THREADS",          # MKL（Intel Mac / conda 版 numpy）
    "VECLIB_MAXIMUM_THREADS",   # Apple Accelerate
    "NUMEXPR_NUM_THREADS",      # numexpr（pandas.eval）
)

# 默认占用逻辑核的比例。0.5 = 一半，留一半给系统
DEFAULT_FRACTION = 0.5

_jobs: int | None = None        # 已生效的线程预算


def _sysctl_int(key: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=2)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def cpu_layout() -> dict:
    """逻辑核总数与（Apple Silicon 上的）性能核 / 能效核拆分。"""
    total = os.cpu_count() or 4
    return {
        "total": total,
        # Apple Silicon：perflevel0 = 性能核，perflevel1 = 能效核
        "perf": _sysctl_int("hw.perflevel0.logicalcpu"),
        "eff": _sysctl_int("hw.perflevel1.logicalcpu"),
    }


def _default_jobs(total: int) -> int:
    """默认线程数 = 逻辑核的一半（本机 18 → 9）。

    ── 为什么是「一半」而不是「只用性能核」──────────────────────────────────
    本机是 6 性能核 + 12 能效核的异构布局，一开始想把线程压在 P 核数（6）以内，
    理由是「OpenMP 并行段要等最慢的 E 核」。**实测推翻了这个假设**——
    RandomForest（300 树，20 万 × 10）在本机的耗时：

        线程数   18     12      9      6
        耗时    8.0s  11.0s  14.8s  22.1s

    一路近似线性，E 核实实在在地在出力（树之间是 embarrassingly parallel，
    没有同步屏障，异构惩罚不成立）。所以砍到 6 是白白多花 2.7 倍时间。

    取一半是个折中：功耗大致减半、系统留出余量（跑训练时电脑还能用），
    代价约 1.85 倍时长。要更凉就 `--jobs 4`，要更快就 `--jobs all`。
    """
    return max(2, int(total * DEFAULT_FRACTION))


def resolve_jobs(jobs: int | str | None = None) -> int:
    """把「预算」解析成一个具体的线程数。

    优先级：显式传参 > QUANT_JOBS 环境变量 > 已有的 OMP_NUM_THREADS
            > 默认（逻辑核的一半）

    `jobs` 可以是：
        正整数   直接用
        0 / -1 / "all"   用满所有逻辑核（旧行为）
        None     走上面的回退链
    """
    if jobs is None:
        jobs = os.environ.get("QUANT_JOBS") or os.environ.get("OMP_NUM_THREADS")

    total = os.cpu_count() or 4

    if jobs is None or jobs == "":
        return _default_jobs(total)

    if isinstance(jobs, str):
        s = jobs.strip().lower()
        if s in ("all", "max", "full"):
            return total
        try:
            jobs = int(s)
        except ValueError:
            return _default_jobs(total)

    jobs = int(jobs)
    if jobs <= 0:                       # 0 / -1 = 用满（沿用 sklearn 的约定）
        return total
    return min(jobs, total)


def apply_thread_limits(jobs: int | str | None = None,
                        verbose: bool = False) -> int:
    """设定全局线程预算，返回实际生效的线程数。

    在 numpy 导入前调用最有效（环境变量那一层）；导入后调用也有效，
    因为 threadpoolctl 能改运行时限额。两次调用以后一次为准。
    """
    global _jobs
    n = resolve_jobs(jobs)

    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(n)

    # 兜底：numpy / sklearn 已经导入时，环境变量不再起作用，改运行时限额
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=n)
    except Exception:
        pass

    _jobs = n
    if verbose:
        lay = cpu_layout()
        detail = (f"{lay['total']} 逻辑核"
                  + (f"（{lay['perf']}P + {lay['eff']}E）"
                     if lay["perf"] and lay["eff"] else ""))
        print(f"[compute] 线程预算 {n} / {detail}"
              + ("  ← 用满，机器会较热" if n >= lay["total"] else ""))
    return n


def get_jobs() -> int:
    """当前线程预算，供 sklearn / LightGBM 的 n_jobs 使用。

    未显式设定过时按默认（逻辑核一半）解析，绝不返回 -1。
    """
    return _jobs if _jobs is not None else resolve_jobs()
