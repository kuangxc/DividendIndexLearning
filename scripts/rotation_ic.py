# -*- coding: utf-8 -*-
"""
信号预测力检验(IC): t 日相对估值 z-score vs 未来 20/60/120 日相对收益(A-B)
若 IC 显著为负 -> "贵后跑输" -> 轮动有依据; 若≈0 -> 相对估值不预测相对走势, 轮动无效
"""
import pandas as pd, numpy as np

CSV = "/Users/kuang.xc/WorkBuddy/2026-06-20-00-20-53/DividendIndexLearning/data/historical.csv"
df = pd.read_csv(CSV, parse_dates=["date"])
names = {"000922":"CSI","000015":"SSE","H30269":"LOWVOL","399324":"SZSE"}
df["idx"] = df["index_code"].astype(str).map(names).fillna(df["index_code"].astype(str))
close = df.pivot(index="date", columns="idx", values="close").sort_index()
pe    = df.pivot(index="date", columns="idx", values="pe_ttm").sort_index()
dy    = df.pivot(index="date", columns="idx", values="dividend_yield").sort_index()

def zscore_ratio(m, a, b, win=250):
    x = np.log(m[a]/m[b])
    mu = x.rolling(win, min_periods=150).mean()
    sd = x.rolling(win, min_periods=150).std()
    return (x-mu)/sd

def fwd_rel(a, b, h):
    return (close[a].shift(-h)/close[a]) - (close[b].shift(-h)/close[b])

pairs = [("SSE","CSI"),("CSI","LOWVOL"),("SSE","LOWVOL")]
print(f"{'对':<14}{'信号':<6}{'期限':>5}{'IC':>8}{'样本':>6}{'|IC|>0.05?':>10}")
for a,b in pairs:
    for m,lbl in [(pe,"PE"),(dy,"DY")]:
        z = zscore_ratio(m,a,b)
        for h in [20,60,120]:
            fr = fwd_rel(a,b,h)
            d = pd.concat([z,fr],axis=1,keys=["z","f"]).dropna()
            ic = d["z"].corr(d["f"])
            flag = "YES" if abs(ic)>0.05 else "no"
            print(f"{a}/{b:<10}{lbl:<6}{h:>5}d{ic:>8.3f}{len(d):>6}{flag:>10}")

# 分桶: z 分档后的未来60日相对收益(更直观)
print("\n=== PE z-score 分档 -> 未来60日相对收益(A-B, pp) ===")
for a,b in pairs:
    z = zscore_ratio(pe,a,b); fr = fwd_rel(a,b,60)
    d = pd.concat([z,fr],axis=1,keys=["z","f"]).dropna()
    bins = pd.cut(d["z"],[-9,-1.5,-0.75,0.75,1.5,9],labels=["<-1.5","-1.5~-0.75","-0.75~0.75","0.75~1.5",">1.5"])
    g = d.groupby(bins, observed=True)["f"].agg(["mean","count"])
    print(f"-- {a}/{b} (正=A跑赢B) --")
    for i,row in g.iterrows():
        print(f"   z {str(i):<12} 均值{row['mean']*100:>6.2f}pp  n={int(row['count'])}")
