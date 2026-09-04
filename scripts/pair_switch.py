import pandas as pd, numpy as np
df = pd.read_csv('/Users/kuang.xc/WorkBuddy/2026-06-20-00-20-53/DividendIndexLearning/data/historical.csv')
df['date'] = pd.to_datetime(df['date'])
p = df.pivot_table(index='date', columns='index_name', values='close').sort_index()
p = p[['上证红利','中证红利']].dropna()
ratio = p['上证红利'] / p['中证红利']

print("=== 价格比（上证红利/中证红利）平稳性 ===")
print(f"均值 {ratio.mean():.4f}  std {ratio.std():.4f}（仅围绕均值波动 {ratio.std()/ratio.mean()*100:.1f}%）")
print(f"区间 {ratio.min():.4f} ~ {ratio.max():.4f}")
x = ratio.values; b = np.polyfit(x[:-1], x[1:], 1)[0]
hl = -np.log(2)/np.log(b) if 0 < b < 1 else np.inf
print(f"AR(1) {b:.3f}  半衰期≈{hl:.0f}天（{'均值回归弱/有漂移' if b>0.985 else '均值回归较快'}）")

def backtest(win, thr, cost=0.0002):
    ma = ratio.rolling(win).mean(); sd = ratio.rolling(win).std()
    z = (ratio - ma)/sd
    raw = pd.Series(np.nan, index=p.index)
    raw[z > thr] = 1.0    # 上证相对贵 → 持中证
    raw[z < -thr] = -1.0  # 上证相对便宜 → 持上证
    pos = raw.ffill().fillna(1.0)
    ret = p.pct_change().shift(-1)
    s = pd.Series(np.where(pos>0, ret['中证红利'], ret['上证红利']), index=p.index)
    sw = (pos != pos.shift()).astype(float)
    s = s - sw*cost
    nav = (1+s.fillna(0)).cumprod()
    yrs = len(p)/252
    ann = (nav.iloc[-1])**(1/yrs)-1
    return (nav.iloc[-1]-1)*100, ann*100, int(sw.sum())

print("\n=== 换仓策略 vs 买入持有 ===")
for c in ['中证红利','上证红利']:
    bh = p[c]/p[c].iloc[0]; yrs=len(p)/252
    print(f"买入持有 {c}: 累计 {(bh.iloc[-1]-1)*100:.1f}%  年化 {((bh.iloc[-1])**(1/yrs)-1)*100:.1f}%")
for win,thr in [(60,0.5),(120,1.0),(250,1.0)]:
    tot,ann,nsw = backtest(win,thr)
    print(f"换仓(窗{win}/阈{thr}σ): 累计 {tot:.1f}%  年化 {ann:.1f}%  换手 {nsw}次")
