# =============================================================================
# ml/nn.py
# 前馈神经网络 NN1 ~ NN5，按 Gu, Kelly & Xiu（2020）附录 B / 表 A.5 实现（待办 #38，2026-10-06）
#
#   结构      隐藏层（ReLU）                 来源
#   NN1       32                             GKX：几何金字塔，每层神经元减半
#   NN2       32-16
#   NN3       32-16-8                        GKX 美股最优
#   NN4       32-16-8-4
#   NN5       32-16-8-4-2
#
# 训练细节（均按 GKX）：
#   - 每个隐藏层：线性 → 批标准化（batch normalization）→ ReLU；输出层线性
#   - 目标：MSE + λ1·Σ|W|（L1 只罚权重，不罚偏置与批标准化参数）
#   - 优化：Adam；批大小 10,000；最多 100 轮
#   - 早停：每轮在**验证集**上算损失，连续 5 轮不改善即停，回滚到最优轮的参数
#   - 集成：10 个随机种子各训一个网络，预测取平均（降低初始化带来的方差）
#   - 超参网格：λ1 ∈ {1e-5, 1e-3} × 学习率 ∈ {0.001, 0.01}
#
# 与其他模型的区别：验证集同时用于早停与选超参，因此**不再合并 train+val 重训**
# （重训就没有早停依据）。GKX 原文即如此；见 models.fit_with_validation 的 early_stopping 分支。
#
# 只用 CPU：MPS / GPU 的浮点运算不保证逐位可复现，而本框架所有结果都要求同输入可复现。
# =============================================================================

from __future__ import annotations

import numpy as np

HIDDEN = {
    "NN1": (32,),
    "NN2": (32, 16),
    "NN3": (32, 16, 8),
    "NN4": (32, 16, 8, 4),
    "NN5": (32, 16, 8, 4, 2),
}


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _train_one(X, y, Xv, yv, hidden, l1, lr, batch_size, max_epochs, patience, seed):
    """训练单个网络，返回最优轮的 state_dict（numpy 化以便跨进程传回）。

    所有原生线程池（OpenBLAS / OpenMP / torch）一律限 1 线程。2026-10-06 实测：conda-forge 的 torch
    与 OpenBLAS 共用环境里的 libomp；若 torch 早于 `src` 加载，`apply_thread_limits` 会把 OpenBLAS /
    OpenMP 设为 9 线程，与 torch 的 1 线程互相冲突，单网络 3 轮从 0.6s 变成 > 40s。
    在此显式限定后与导入顺序无关；并行只发生在种子之间（GKXNet.fit）。
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _train_one_impl(X, y, Xv, yv, hidden, l1, lr, batch_size, max_epochs, patience, seed)


def _train_one_impl(X, y, Xv, yv, hidden, l1, lr, batch_size, max_epochs, patience, seed):
    import torch
    from torch import nn

    torch.set_num_threads(1)                 # 并行发生在种子之间，单网络单线程
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    net = nn.Sequential(*layers)
    weights = [m.weight for m in net if isinstance(m, nn.Linear)]
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # joblib 以只读内存映射传入大数组；torch 要求可写 → 先复制（训练中不写数据，复制只为消除未定义行为）
    Xt, yt = torch.from_numpy(np.array(X)), torch.from_numpy(np.array(y))
    Xvt, yvt = torch.from_numpy(np.array(Xv)), torch.from_numpy(np.array(yv))
    n = len(Xt)
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        net.train()
        for idx in np.array_split(rng.permutation(n), max(1, n // batch_size)):
            ii = torch.from_numpy(idx)
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt[ii]).squeeze(1), yt[ii])
            loss = loss + l1 * sum(w.abs().sum() for w in weights)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(Xvt).squeeze(1), yvt).item()
        if v < best - 1e-12:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return {k: t.numpy() for k, t in best_state.items()}


class GKXNet:
    """GKX 式前馈网络（sklearn 风格：fit / predict），10 种子集成。"""

    def __init__(self, hidden=(32, 16, 8), l1=1e-5, lr=1e-3, batch_size=10_000,
                 max_epochs=100, patience=5, n_ensemble=10, seed=0, n_jobs=None):
        self.hidden, self.l1, self.lr = tuple(hidden), l1, lr
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.n_ensemble, self.seed, self.n_jobs = n_ensemble, seed, n_jobs

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None:
            raise ValueError("[nn] GKXNet 需要验证集做早停（fit(X, y, X_val, y_val)）")
        from joblib import Parallel, delayed, parallel_config
        from src.config.compute import get_jobs

        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
        X, y, Xv, yv = f32(X), f32(y), f32(X_val), f32(y_val)
        jobs = self.n_jobs or get_jobs()
        # 并行发生在种子之间；子进程内部限 1 线程。否则子进程继承 OMP_NUM_THREADS（src/__init__ 设为线程预算），
        # 9 个进程 × 9 个 OpenMP 线程超订 CPU，实测慢一个数量级以上（2026-10-06）
        with parallel_config(backend="loky", inner_max_num_threads=1):
            self.states_ = Parallel(n_jobs=min(jobs, self.n_ensemble))(
                delayed(_train_one)(X, y, Xv, yv, self.hidden, self.l1, self.lr, self.batch_size,
                                    self.max_epochs, self.patience, self.seed + k)
                for k in range(self.n_ensemble))
        self.n_features_ = X.shape[1]
        return self

    def _net(self):
        from torch import nn
        layers, d = [], self.n_features_
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def predict(self, X):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):          # 同 _train_one：避免线程池冲突
            return self._predict(X)

    def _predict(self, X):
        import torch
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        out = np.zeros(len(Xt))
        for st in self.states_:
            net = self._net()
            net.load_state_dict({k: torch.from_numpy(v) for k, v in st.items()})
            net.eval()
            with torch.no_grad():
                out += net(Xt).squeeze(1).double().numpy()
        return out / len(self.states_)
