#!/usr/bin/env python3
"""
补齐 dividend_yield 历史数据（2020 年起）。

策略：
1. 用中证官网 /csindex-home/perf/indexCsiDsPe 获取历史 PE-TTM（可覆盖 2020 至今）
2. 用 stock_zh_index_value_csindex 获取近期股息率（近 20 个交易日）
3. 计算近期股息支付率 = dividend_yield * PE_TTM
4. 假设股息支付率相对稳定，用最近 N 天的平均股息支付率反推历史 dividend_yield = payout_ratio / PE_TTM
5. 保留已有近期真实/补充股息率，只填补缺失值

注意：这是一个估算值，不是精确值。在 data.md 中标注为"估算"。
"""
import time
import argparse
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import akshare as ak

REPO = Path(__file__).resolve().parent.parent
HIST_CSV = REPO / "data" / "historical.csv"

INDICES = {
    "000922": "000922",
    "000015": "000015",
    "H30269": "H30269",
    "399324": "399324",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_payout_ratio_from_recent(code_csindex, days=20):
    """
    用近期数据计算平均股息支付率。
    payout_ratio = dividend_yield * PE_TTM
    """
    try:
        df = ak.stock_zh_index_value_csindex(symbol=code_csindex)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').tail(days)
        df['股息率2'] = pd.to_numeric(df['股息率2'], errors='coerce')
        df['市盈率2'] = pd.to_numeric(df['市盈率2'], errors='coerce')
        # 去除异常值
        mask = df['股息率2'].notna() & df['市盈率2'].notna() & (df['市盈率2'] > 0)
        sub = df[mask].copy()
        if sub.empty:
            return None
        sub['payout_ratio'] = sub['股息率2'] / 100 * sub['市盈率2']  # 股息率2 是百分比
        payout = sub['payout_ratio'].median()
        log(f"  近期股息支付率中位数: {payout:.4f} (基于 {len(sub)} 个数据点)")
        return payout
    except Exception as e:
        log(f"  计算股息支付率失败: {e}")
        return None


def fetch_historical_pe(code_csindex, start_date, end_date):
    """从中证官网获取历史 PE-TTM。

    注意：`peg` 是中证官网该历史估值接口返回的字段名，实测对应指数滚动市盈率数据；
    该接口不返回股息率，仅用于估算股息率时提供历史 PE。
    """
    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")
    url = "https://www.csindex.com.cn/csindex-home/perf/indexCsiDsPe"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.csindex.com.cn/",
        "Accept": "application/json,text/plain,*/*",
    }
    params = {"indexCode": code_csindex, "startDate": start, "endDate": end}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        if payload.get("code") != "200" or not data:
            log(f"  中证历史 PE 返回为空: {payload.get('msg')}")
            return pd.DataFrame(columns=["日期", "滚动市盈率"])

        df = pd.DataFrame(data)
        df = df.rename(columns={"tradeDate": "日期", "peg": "滚动市盈率"})
        df["日期"] = pd.to_datetime(df["日期"], format="%Y%m%d", errors="coerce")
        df["滚动市盈率"] = pd.to_numeric(df["滚动市盈率"], errors="coerce")
        df = df.dropna(subset=["日期", "滚动市盈率"])
        df = df.drop_duplicates(subset=["日期"], keep="last")
        mask = (df["日期"] >= pd.to_datetime(start_date)) & (df["日期"] <= pd.to_datetime(end_date))
        return df.loc[mask, ["日期", "滚动市盈率"]].sort_values("日期").reset_index(drop=True)
    except Exception as e:
        log(f"  获取中证历史 PE 失败: {e}")
        return pd.DataFrame(columns=["日期", "滚动市盈率"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="处理所有指数")
    args = parser.parse_args()

    if not HIST_CSV.exists():
        log(f"historical.csv not found: {HIST_CSV}")
        return

    df = pd.read_csv(HIST_CSV, parse_dates=["date"])
    log(f"Loaded {len(df)} rows from historical.csv")

    for code, code_csindex in INDICES.items():
        log(f"Processing {code}...")
        sub = df[(df["index_code"] == code) & (df["dividend_yield"].isna())]
        if sub.empty:
            log(f"  {code}: all dividend_yield already filled, skipping")
            continue

        log(f"  {code}: need to fill {len(sub)} rows")

        # 获取近期股息支付率
        payout_ratio = get_payout_ratio_from_recent(code_csindex)
        if payout_ratio is None:
            log(f"  {code}: cannot compute payout ratio, skipping")
            continue

        # 获取历史 PE
        hist_pe = fetch_historical_pe(code_csindex, "2020-01-01", "2026-06-20")
        if hist_pe.empty:
            log(f"  {code}: no historical PE data, skipping")
            continue

        log(f"  got {len(hist_pe)} rows of historical PE")

        # 反推 dividend_yield = payout_ratio / PE_TTM * 100 (转成百分比)
        hist_pe['estimated_dy'] = hist_pe['滚动市盈率'].apply(
            lambda pe: round(payout_ratio / pe * 100, 4) if pd.notna(pe) and pe > 0 else None
        )

        # 合并到主 df
        for _, row in hist_pe.iterrows():
            d = row['日期']
            mask = (df["index_code"] == code) & (df["date"] == d)
            if mask.any() and pd.isna(df.loc[mask, "dividend_yield"].values[0]):
                df.loc[mask, "dividend_yield"] = row['estimated_dy']

        filled = df[(df["index_code"] == code) & df["dividend_yield"].notna()].shape[0]
        log(f"  {code}: total filled rows now = {filled}")

        # 避免请求过快
        time.sleep(1)

    # Save
    df.to_csv(HIST_CSV, index=False, float_format="%.4f")
    log(f"Saved to {HIST_CSV}")

    # 打印覆盖情况
    final = df[df["dividend_yield"].notna()].groupby("index_code")["date"].agg(["min", "max", "count"])
    print("\n=== dividend_yield coverage after update ===")
    print(final.to_string())


if __name__ == "__main__":
    main()
