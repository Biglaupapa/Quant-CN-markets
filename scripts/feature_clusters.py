"""
特征聚类（López de Prado 2020《Machine Learning for Asset Managers》ONC 的基础聚类步骤）+ 与 GKX 四大类交叉对照。

- 相关：112 个 ML 特征两两的月均截面 Spearman（2007-01 ~ 2026-08，cache 因子值）
- 距离：d = sqrt(1 − |ρ|)。取绝对值：因子正负号只是定义方向（dolvol_126 与 amihud ρ = −0.90 实为同类替代）。
  LdP 原式为 sqrt((1 − ρ)/2)，此处为适配「方向任意」的偏离，已记录
- 聚类：完整 ONC（LdP 2020 第 4 章 clusterKMeansBase + clusterKMeansTop）：
  以距离矩阵的行为观测做 KMeans，k = 2..n−1、重复 n_init 次，取质量分 q = mean(silh)/std(silh) 最大者；
  再对「组内质量分低于平均」的组在其子相关矩阵上递归重聚，仅当新的平均组质量提高时才采纳
  （2026-10-03 初版只做了基础一步，k 上限 20，结果仅 2 组、轮廓 0.18，弃用）
- 输出：docs/reference/feature_clusters.csv，及 GKX 类别 × 聚类组 交叉表、ARI
用法：python scripts/feature_clusters.py
"""
import sys; sys.path.insert(0, "/Users/louis/MyProjects/Quant")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, adjusted_rand_score

G = pd.read_csv("docs/reference/feature_groups_gkx.csv")
G["GKX组"] = np.where(G["GKX类别"].str.startswith("4"), np.where(G["会计块"], "4b 会计", "4a 估值"), G["GKX类别"])
feats = G["特征"].tolist()
F = {f: (lambda x: x.set_axis(pd.to_datetime(x.index)))(pd.read_csv(f"output/cache/{f}.csv", index_col=0)).loc["2007":"2026-08"]
     for f in feats}
months = sorted(set.intersection(*[set(v.index) for v in F.values()]))
acc, cnt = np.zeros((len(feats), len(feats))), np.zeros((len(feats), len(feats)))
for m in months:
    X = pd.DataFrame({f: F[f].loc[m] for f in feats}).rank()
    c = X.corr(min_periods=200).to_numpy()
    ok = ~np.isnan(c); acc[ok] += c[ok]; cnt[ok] += 1
r = acc / np.maximum(cnt, 1)
np.fill_diagonal(r, 1.0)
rho = pd.DataFrame(r, index=feats, columns=feats)


def _dist(c: pd.DataFrame) -> pd.DataFrame:
    return ((1 - c.abs()) / 2.0).clip(lower=0) ** 0.5          # LdP：((1−ρ)/2)^½，ρ 取绝对值


def base(corr: pd.DataFrame, max_k: int, n_init: int = 10):
    x = _dist(corr)
    best = None
    for init in range(n_init):
        for k in range(2, max_k + 1):
            lab = KMeans(n_clusters=k, n_init=1, random_state=init * 1000 + k).fit_predict(x)
            sl = silhouette_samples(x, lab)
            q = sl.mean() / sl.std()
            if best is None or q > best[0]:
                best = (q, lab, sl)
    lab, sl = best[1], pd.Series(best[2], index=corr.index)
    cl = {i: corr.index[lab == i].tolist() for i in np.unique(lab)}
    return cl, sl


def _quality(sl: pd.Series, cl: dict) -> dict:
    return {i: sl[m].mean() / sl[m].std() if len(m) > 1 and sl[m].std() > 0 else 0.0 for i, m in cl.items()}


def top(corr: pd.DataFrame, max_k=None, n_init: int = 10):
    max_k = min(max_k or corr.shape[1] - 1, corr.shape[1] - 1)
    if max_k < 2:
        return {0: corr.index.tolist()}, pd.Series(0.0, index=corr.index)
    cl, sl = base(corr, max_k, n_init)
    qs = _quality(sl, cl)
    mean_q = np.mean(list(qs.values()))
    redo = [i for i, q in qs.items() if q < mean_q]
    if len(redo) <= 1:
        return cl, sl
    keys = [f for i in redo for f in cl[i]]
    cl2, _ = top(corr.loc[keys, keys], max_k=min(max_k, len(keys) - 1), n_init=n_init)
    new = {j: m for j, m in enumerate([cl[i] for i in cl if i not in redo] + list(cl2.values()))}
    lab = pd.Series(0, index=corr.index)
    for j, m in new.items():
        lab[m] = j
    x = _dist(corr)
    sl_new = pd.Series(silhouette_samples(x, lab.loc[corr.index].to_numpy()), index=corr.index)
    if np.mean(list(_quality(sl_new, new).values())) <= np.mean([qs[i] for i in redo]):
        return cl, sl
    return new, sl_new

clusters, silh = top(rho, n_init=10)
k = len(clusters)
lab = pd.Series(0, index=feats)
for j, m in enumerate(sorted(clusters.values(), key=len, reverse=True)):
    lab[m] = j
lab = lab.to_numpy()
q = np.mean(list(_quality(silh, {i: [feats[t] for t in np.where(lab == i)[0]] for i in range(k)}).values()))
smean = silh.mean()
G["聚类组"] = [f"C{l+1}" for l in lab]
G.to_csv("docs/reference/feature_clusters.csv", index=False)
rho.to_csv("docs/reference/feature_corr_xs_spearman.csv")
print(f"ONC 组数 k = {k}，平均组质量分 {q:.3f}，平均轮廓系数 {smean:.3f}，月数 {len(months)}")
ct = pd.crosstab(G["聚类组"], G["GKX组"]); ct = ct.loc[sorted(ct.index, key=lambda x: int(x[1:]))]
print("\nGKX 组 × 聚类组 交叉表：\n", ct.to_string())
print("\nARI（调整兰德指数，1 = 完全一致，0 = 随机）:", round(adjusted_rand_score(G["GKX组"], G["聚类组"]), 3))
pur = ct.max(axis=1).sum() / ct.values.sum()
print("纯度（每个聚类组中占多数的 GKX 组所占比例，加总）:", round(pur, 3))
for c, g in G.groupby("聚类组"):
    print(f"\n{c}（{len(g)}）:", ", ".join(g["特征"]) if len(g) <= 25 else ", ".join(g["特征"].head(25)) + f" …（另 {len(g)-25} 个）")
