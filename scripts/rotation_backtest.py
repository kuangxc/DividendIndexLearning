# -*- coding: utf-8 -*-
"""
红利指数估值轮动回测
信号: PE/PB/DY 的相对差值(比值 z-score / 绝对差), 滞回带防抖动
基准: 买入持有各腿 + 50/50 再平衡
成本: 单边 0.05% (ETF 佣金+冲击, 保守)
无未来函数: t 日信号 -> t+1 日收益
"""
import pandas as pd, numpy as np, itertools

CSV = "/Users/kuang.xc/WorkBuddy/2026-06-20-00-20-53/DividendIndexLearning/data/historical.csv"
COST = 0.0005  # 单边万5

df = pd.read_csv(CSV, parse_dates=["date"])
names = {"000922":"CSI_DIV","000015":"SSE_DIV","H30269":"DIV_LOWVOL","399324":"SZSE_DIV"}
df["idx"] = df["index_code"].astype(str).map(names).fillna(df["index_code"].astype(str))

close = df.pivot(index="date", columns="idx", values="close").sort_index()
pe    = df.pivot(index="date", columns="idx", values="pe_ttm").sort_index()
pb    = df.pivot(index="date", columns="idx", values="pb").sort_index()
dy    = df.pivot(index="date", columns="idx", values="dividend_yield").sort_index()

print("=== 数据覆盖(NaN 检查) ===")
for m, t in [("close",close),("pe",pe),("pb",pb),("dy",dy)]:
    nn = t.notna().sum()
    print(f"{m}: " + ", ".join(f"{c}:{nn[c]}" for c in t.columns))

ret = close.pct_change().shift(-1)   # t 日收盘买入 -> t+1 收益, 用 shift(-1) 对齐信号日
# 注意: ret.loc[t] 表示 t->t+1 的收益

def perf(pos, rets, legs):
    """pos: Series of leg names (t 日持有的腿), rets: DataFrame t->t+1 收益"""
    legs = list(legs)
    pos = pos.reindex(rets.index).ffill()
    r = pd.Series(index=rets.index, dtype=float)
    prev = None; switches = 0
    for d in rets.index:
        cur = pos.loc[d]
        if pd.isna(cur): r.loc[d] = np.nan; continue
        base = rets.loc[d, cur]
        cost = 0.0
        if prev is not None and cur != prev:
            cost = 2*COST; switches += 1
        r.loc[d] = base - cost if pd.notna(base) else np.nan
        prev = cur
    r = r.dropna()
    if len(r) < 30: return None
    nav = (1+r).cumprod()
    years = len(r)/244
    ann = nav.iloc[-1]**(1/years)-1
    mdd = (nav/nav.cummax()-1).min()
    return dict(total=nav.iloc[-1]-1, ann=ann, mdd=mdd, switches=switches, nav=nav)

def bh(leg, rets):
    r = rets[leg].dropna()
    nav = (1+r).cumprod(); years=len(r)/244
    return dict(total=nav.iloc[-1]-1, ann=nav.iloc[-1]**(1/years)-1, mdd=(nav/nav.cummax()-1).min(), switches=0, nav=nav)

def rb5050(a, b, rets):
    r = (rets[a]+rets[b])/2
    r = r.dropna()
    nav=(1+r).cumprod(); years=len(r)/244
    return dict(total=nav.iloc[-1]-1, ann=nav.iloc[-1]**(1/years)-1, mdd=(nav/nav.cummax()-1).min(), switches=0, nav=nav)

# ---------- 信号生成 ----------
def sig_zscore_ratio(m, a, b, win=250, band=0.75, log=True):
    """m=a/b 的 z-score 滞回带: z>band 持 b(a贵), z<-band 持 a, 其间保持"""
    ratio = m[a]/m[b]
    x = np.log(ratio) if log else ratio
    mu = x.rolling(win, min_periods=int(win*0.6)).mean()
    sd = x.rolling(win, min_periods=int(win*0.6)).std()
    z = (x-mu)/sd
    pos = pd.Series(index=m.index, dtype=object); cur=None
    for d in m.index:
        zz = z.loc[d]
        if pd.notna(zz):
            if zz > band: cur = b
            elif zz < -band: cur = a
        pos.loc[d] = cur
    return pos, z

def sig_dy_hyst(a, b, hyst=0.20):
    """持有 DY 更高者, 仅当另一腿 DY 超出当前腿 hyst 个百分点才换"""
    pos = pd.Series(index=dy.index, dtype=object); cur=None
    for d in dy.index:
        da, db = dy.loc[d,a], dy.loc[d,b]
        if pd.isna(da) or pd.isna(db): pos.loc[d]=cur; continue
        if cur is None: cur = a if da>=db else b
        elif cur==a and db > da+hyst: cur=b
        elif cur==b and da > db+hyst: cur=a
        pos.loc[d]=cur
    return pos

def sig_pct_rank(a, b, win=750, hyst=15):
    """持有自身 PE 分位更低者, 分位差>hyst 个百分点才换"""
    pa = pe[a].rolling(win, min_periods=250).apply(lambda s: (s.iloc[-1]>=s).mean()*100, raw=False)
    pbb = pe[b].rolling(win, min_periods=250).apply(lambda s: (s.iloc[-1]>=s).mean()*100, raw=False)
    pos = pd.Series(index=pe.index, dtype=object); cur=None
    for d in pe.index:
        ra, rb_ = pa.loc[d], pbb.loc[d]
        if pd.isna(ra) or pd.isna(rb_): pos.loc[d]=cur; continue
        if cur is None: cur = a if ra<=rb_ else b
        elif cur==a and rb_ < ra-hyst: cur=b
        elif cur==b and ra < rb_-hyst: cur=a
        pos.loc[d]=cur
    return pos

# ---------- 逐对回测 ----------
pairs = [("SSE_DIV","CSI_DIV"),("CSI_DIV","DIV_LOWVOL"),("SSE_DIV","DIV_LOWVOL")]
rows = []
for a,b in pairs:
    rets = ret[[a,b]]
    res = {}
    res[f"BH_{a}"] = bh(a, rets); res[f"BH_{b}"] = bh(b, rets)
    res["50/50"] = rb5050(a,b,rets)
    for label, sig in [
        ("PE_z(250,0.75)", sig_zscore_ratio(pe,a,b,250,0.75)[0]),
        ("PB_z(250,0.75)", sig_zscore_ratio(pb,a,b,250,0.75)[0]),
        ("DY_z(250,0.75)", sig_zscore_ratio(dy,a,b,250,0.75,log=True)[0]),
        ("PE_z(250,1.0)",  sig_zscore_ratio(pe,a,b,250,1.0)[0]),
        ("DY高(滞回0.2pp)", sig_dy_hyst(a,b,0.20)),
        ("PE分位低(滞回15)", sig_pct_rank(a,b)),
    ]:
        p = perf(sig, rets, [a,b])
        if p: res[label]=p
    print(f"\n===== {a} <-> {b} =====")
    print(f"{'策略':<20}{'累计':>9}{'年化':>8}{'最大回撤':>9}{'换手':>6}")
    for k,v in res.items():
        print(f"{k:<20}{v['total']*100:>8.1f}%{v['ann']*100:>7.2f}%{v['mdd']*100:>8.1f}%{v['switches']:>6}")
    best = max([k for k in res if k not in (f"BH_{a}",f"BH_{b}","50/50")], key=lambda k: res[k]['ann'])
    better = max(res[f"BH_{a}"]['ann'], res[f"BH_{b}"]['ann'], res["50/50"]['ann'])
    rows.append((a,b,best,res[best]['ann'],res[best]['switches'],better,res[best]['ann']-better))

# ---------- 三选一轮动: 中证红利/红利低波/上证红利 ----------
legs3 = ["CSI_DIV","DIV_LOWVOL","SSE_DIV"]
rets3 = ret[legs3]
def sig3_dy(hyst=0.20, min_hold=20):
    pos = pd.Series(index=dy.index, dtype=object); cur=None; hold=0
    for d in dy.index:
        v = dy.loc[d, legs3]
        if v.isna().any(): pos.loc[d]=cur; continue
        best = v.idxmax()
        if cur is None: cur=best; hold=0
        else:
            hold+=1
            if hold>=min_hold and v[best] > v[cur]+hyst:
                cur=best; hold=0
        pos.loc[d]=cur
    return pos
def sig3_pct(win=750, hyst=15, min_hold=20):
    pct = {L: pe[L].rolling(win, min_periods=250).apply(lambda s:(s.iloc[-1]>=s).mean()*100, raw=False) for L in legs3}
    pos = pd.Series(index=pe.index, dtype=object); cur=None; hold=0
    for d in pe.index:
        v = pd.Series({L: pct[L].loc[d] for L in legs3})
        if v.isna().any(): pos.loc[d]=cur; continue
        best = v.idxmin()
        if cur is None: cur=best; hold=0
        else:
            hold+=1
            if hold>=min_hold and v[best] < v[cur]-hyst:
                cur=best; hold=0
        pos.loc[d]=cur
    return pos

print("\n===== 三选一: CSI_DIV / DIV_LOWVOL / SSE_DIV =====")
res3 = {"BH_CSI":bh("CSI_DIV",rets3), "BH_LOWVOL":bh("DIV_LOWVOL",rets3), "BH_SSE":bh("SSE_DIV",rets3)}
for label,sig in [("DY最高(滞回0.2,持20d)",sig3_dy()), ("PE分位最低(滞回15,持20d)",sig3_pct())]:
    p = perf(sig, rets3, legs3)
    if p: res3[label]=p
print(f"{'策略':<26}{'累计':>9}{'年化':>8}{'最大回撤':>9}{'换手':>6}")
for k,v in res3.items():
    print(f"{k:<26}{v['total']*100:>8.1f}%{v['ann']*100:>7.2f}%{v['mdd']*100:>8.1f}%{v['switches']:>6}")

# ---------- 信号特征的均值回归性质 ----------
print("\n===== 相对估值差值的均值回归特征 =====")
for a,b in pairs:
    for m,lbl in [(pe,"PE"),(pb,"PB"),(dy,"DY")]:
        x = np.log(m[a]/m[b]).dropna()
        if len(x)<300: continue
        ar1 = x.autocorr(1)
        hl = np.log(2)/(-np.log(ar1)) if 0<ar1<1 else np.nan
        z = (x-x.rolling(250,min_periods=150).mean())/x.rolling(250,min_periods=150).std()
        print(f"{a}/{b} {lbl}比: 全期std={x.std():.3f} AR(1)={ar1:.4f} 半衰期={hl:.0f}d |z|>1占比={((z.abs()>1).mean()*100):.1f}%")

print("\n===== 汇总: 各对最优策略 vs 最优基准 =====")
for a,b,best,ann,sw,better,edge in rows:
    print(f"{a}<->{b}: 最优[{best}] 年化{ann*100:.2f}% 换手{sw} | 最优基准年化{better*100:.2f}% | 超额{edge*100:+.2f}pp/年")
