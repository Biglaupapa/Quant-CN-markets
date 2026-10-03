import sys; sys.path.insert(0,"/Users/louis/MyProjects/Quant")
import pandas as pd, numpy as np, glob, os, warnings; warnings.filterwarnings("ignore")
from src.backtest.engine import calc_monthly_returns, group_return
from src.backtest.metrics import calc_ic
from src.strategy.combine_factors import align_factor_directions, combine_factors
from src.main import COMBINE_FACTORS, FACTOR_DIRECTIONS
SP="/private/tmp/claude-501/-Users-louis-MyProjects-Database/02e109d4-c996-432e-ac94-9da81074657d/scratchpad"
IS, OOS = ("2007-01-01","2016-12-31"), ("2017-01-01","2026-08-31")
ret=calc_monthly_returns("2007-01-01","2026-09-30"); fwd=ret.shift(-1)
F={os.path.basename(p)[:-4]: (lambda x: x.set_axis(pd.to_datetime(x.index)))(pd.read_csv(p,index_col=0)) for p in glob.glob("output/cache/*.csv")}
pool=COMBINE_FACTORS["microstructure"]+COMBINE_FACTORS["fundamental"]
def ic_of(f,a,b): x=calc_ic(f,fwd,method="spearman").loc[a:b].dropna(); return x
def stats(comp,a,b):
    ic=ic_of(comp,a,b); g=group_return(comp,fwd,n_groups=5).loc[a:b]; ls=g["LS"].dropna()
    nav=(1+ls).cumprod(); mdd=(nav/nav.cummax()-1).min()
    return dict(IC=ic.mean(),ICIR=ic.mean()/ic.std(),LS年化=ls.mean()*12,夏普=ls.mean()/ls.std()*np.sqrt(12),回撤=mdd)
def composite(names,dirs):
    al=align_factor_directions({n:F[n] for n in names},dirs); return combine_factors(al,weights="equal")
# 样本内单因子统计 + 方向
rows=[]
for n,f in F.items():
    x=ic_of(f,*IS); 
    if len(x)==0: continue
    rows.append(dict(因子=n,IS_IC=x.mean(),IS_ICIR=x.mean()/x.std(),IS月数=len(x)))
uni=pd.DataFrame(rows).set_index("因子"); nmon=uni.IS月数.max()
dirs={n:(FACTOR_DIRECTIONS.get(n) if n in pool else int(np.sign(uni.loc[n,"IS_IC"]))) for n in uni.index}
# 与池内因子的截面秩相关（样本内月均）
def xcorr(a,b):
    A=F[a].loc[IS[0]:IS[1]].rank(axis=1); B=F[b].reindex_like(A).rank(axis=1)
    return A.corrwith(B,axis=1).mean()
cand=uni[(uni.IS_ICIR.abs()>=0.4)&(uni.IS月数>=0.9*nmon)&(~uni.index.isin(pool))].copy()
cand["与池最大|相关|"]=[max(abs(xcorr(c,p)) for p in pool) for c in cand.index]
cand["最相关池因子"]=[max(pool,key=lambda p: abs(xcorr(c,p))) for c in cand.index]
base={k:stats(composite(pool,dirs),*v) for k,v in [("IS",IS),("OOS",OOS)]}
print("基准（现有 10 因子等权）:", {k:{m:round(x,3) for m,x in v.items()} for k,v in base.items()})
res=[]
for c in cand.index:
    s={k:stats(composite(pool+[c],dirs),*v) for k,v in [("IS",IS),("OOS",OOS)]}
    res.append(dict(因子=c,IS_ICIR=cand.loc[c,"IS_ICIR"],最大相关=cand.loc[c,"与池最大|相关|"],最相关=cand.loc[c,"最相关池因子"],
        IS夏普Δ=s["IS"]["夏普"]-base["IS"]["夏普"],OOS夏普Δ=s["OOS"]["夏普"]-base["OOS"]["夏普"],
        OOS_ICIRΔ=s["OOS"]["ICIR"]-base["OOS"]["ICIR"],OOS回撤Δ=s["OOS"]["回撤"]-base["OOS"]["回撤"]))
add=pd.DataFrame(res).set_index("因子").sort_values("IS夏普Δ",ascending=False)
drop=[]
for p in pool:
    rest=[x for x in pool if x!=p]; s={k:stats(composite(rest,dirs),*v) for k,v in [("IS",IS),("OOS",OOS)]}
    drop.append(dict(去掉=p,IS夏普Δ=s["IS"]["夏普"]-base["IS"]["夏普"],OOS夏普Δ=s["OOS"]["夏普"]-base["OOS"]["夏普"],OOS_ICIRΔ=s["OOS"]["ICIR"]-base["OOS"]["ICIR"]))
drop=pd.DataFrame(drop).set_index("去掉").sort_values("IS夏普Δ",ascending=False)
pd.set_option("display.width",220)
print(f"\n候选（样本内 |ICIR|≥0.4、覆盖≥90%、不在池中）共 {len(add)} 个：\n", add.round(3).to_string())
print("\n去掉一个（Δ>0 表示去掉后更好）：\n", drop.round(3).to_string())
add.to_csv(f"{SP}/pool_add.csv"); drop.to_csv(f"{SP}/pool_drop.csv"); uni.to_csv(f"{SP}/pool_uni_is.csv")
