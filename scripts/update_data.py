#!/usr/bin/env python3
"""
红利指数数据采集与绘图脚本
支持：数据回填（2020年起）、增量更新、图表生成

数据源：
- 乐咕乐股(legulegu): 上证红利/深证红利 PE/PB/收盘价 历史（2005年起）
- 中证指数(csindex): 全部指数 近期PE/股息率 + 历史行情
- 东方财富(eastmoney): 全部指数 行情（备选）

使用：
    python scripts/update_data.py --backfill      # 回填历史数据
    python scripts/update_data.py --incremental   # 增量更新（默认）
    python scripts/update_data.py --charts        # 仅生成图表
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Chart text uses English only to avoid font/encoding issues across runners.
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

import akshare as ak

# 配置
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / 'data'
CHARTS_DIR = REPO_ROOT / 'charts'
HISTORICAL_CSV = DATA_DIR / 'historical.csv'

# 指数定义
INDICES = {
    '000922': {
        'name': '中证红利',
        'english_name': 'CSI Dividend',
        'symbol_legulegu': None,  # 乐咕乐股不支持
        'symbol_eastmoney': 'sh000922',
        'symbol_csindex': '000922',
        'start_date': '2020-01-01',
    },
    '000015': {
        'name': '上证红利',
        'english_name': 'SSE Dividend',
        'symbol_legulegu': '上证红利',
        'symbol_eastmoney': 'sh000015',
        'symbol_csindex': '000015',
        'start_date': '2020-01-01',
    },
    '399324': {
        'name': '深证红利',
        'english_name': 'SZSE Dividend',
        'symbol_legulegu': '深证红利',
        'symbol_eastmoney': 'sz399324',
        'symbol_csindex': '399324',
        'start_date': '2020-01-01',
    },
    'H30269': {
        'name': '红利低波',
        'english_name': 'Dividend Low Vol',
        'symbol_legulegu': None,  # 乐咕乐股不支持
        'symbol_eastmoney': 'shH30269',
        'symbol_csindex': 'H30269',
        'start_date': '2020-01-01',
    },
}

# 国债收益率默认值
BOND_YIELD_DEFAULT = 1.8


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def format_date_axis(ax, dates):
    """Format date ticks without shrinking the chart's history range."""
    valid_dates = pd.to_datetime(pd.Series(dates), errors='coerce').dropna()
    if valid_dates.empty:
        return

    min_date = valid_dates.min()
    max_date = valid_dates.max()
    span_days = max((max_date - min_date).days, 1)

    ax.set_xlim(min_date, max_date)
    if span_days <= 180:
        interval = max(span_days // 8, 1)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif span_days <= 730:
        interval = max(span_days // 240, 1)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def load_existing_data():
    """加载已有历史数据"""
    if HISTORICAL_CSV.exists():
        df = pd.read_csv(HISTORICAL_CSV, parse_dates=['date'])
        log(f"加载已有数据: {len(df)} 行, 日期范围 {df['date'].min().date()} ~ {df['date'].max().date()}")
        return df
    else:
        log("历史数据文件不存在，创建新数据")
        return pd.DataFrame(columns=[
            'date', 'index_code', 'index_name', 'close',
            'pe_ttm', 'pe_static', 'pb', 'dividend_yield', 'bond_yield'
        ])


def fetch_eastmoney_history(symbol, start_date, end_date):
    """从东方财富获取指数历史行情（备选）"""
    try:
        df = ak.stock_zh_index_daily_em(
            symbol=symbol,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
        )
        if df.empty:
            return None
        df = df.rename(columns={'date': 'date'})[['date', 'close']]
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        log(f"  东方财富行情获取失败 {symbol}: {e}")
        return None


def fetch_legulegu_history(symbol, start_date, end_date):
    """从乐咕乐股获取PE历史（含指数点位/收盘价）"""
    if not symbol:
        return None
    try:
        df = ak.stock_index_pe_lg(symbol=symbol)
        df = df.rename(columns={
            '日期': 'date',
            '指数': 'close',
            '滚动市盈率': 'pe_ttm',
            '静态市盈率': 'pe_static',
        })[['date', 'close', 'pe_ttm', 'pe_static']]
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        return df[mask].copy()
    except Exception as e:
        log(f"  乐咕乐股数据获取失败 {symbol}: {e}")
        return None


def fetch_legulegu_pb(symbol, start_date, end_date):
    """从乐咕乐股获取PB历史"""
    if not symbol:
        return None
    try:
        df = ak.stock_index_pb_lg(symbol=symbol)
        df = df.rename(columns={
            '日期': 'date',
            '市净率': 'pb',
        })[['date', 'pb']]
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        return df[mask].copy()
    except Exception as e:
        log(f"  乐咕乐股PB获取失败 {symbol}: {e}")
        return None


def fetch_csindex_history(symbol, start_date, end_date):
    """从中证指数获取历史行情（收盘价+滚动市盈率）"""
    try:
        df = ak.stock_zh_index_hist_csindex(
            symbol=symbol,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
        )
        if df.empty:
            return None
        cols = ['日期', '收盘']
        rename = {'日期': 'date', '收盘': 'close'}
        # 如果有滚动市盈率，一并提取
        if '滚动市盈率' in df.columns:
            cols.append('滚动市盈率')
            rename['滚动市盈率'] = 'pe_ttm'
        df = df[cols].rename(columns=rename)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        log(f"  中证历史行情获取失败 {symbol}: {e}")
        return None


def fetch_csindex_valuation(symbol, start_date, end_date):
    """从中证指数获取PE/股息率"""
    try:
        df = ak.stock_zh_index_value_csindex(symbol=symbol)
        df = df.rename(columns={
            '日期': 'date',
            '市盈率1': 'pe_static',
            '市盈率2': 'pe_ttm',
            '股息率1': 'dividend_yield',
        })[['date', 'pe_static', 'pe_ttm', 'dividend_yield']]
        df['date'] = pd.to_datetime(df['date'])
        df['dividend_yield'] = pd.to_numeric(df['dividend_yield'], errors='coerce')
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        return df[mask].copy()
    except Exception as e:
        log(f"  中证指数估值获取失败 {symbol}: {e}")
        return None


def fetch_bond_yield():
    """获取10年期国债收益率"""
    try:
        # 尝试akshare的债券接口
        for func_name in ['bond_zh_yield', 'bond_china_yield']:
            if hasattr(ak, func_name):
                func = getattr(ak, func_name)
                df = func(start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                         end_date=datetime.now().strftime('%Y%m%d'))
                if '国债收益率10年' in df.columns:
                    latest = df['国债收益率10年'].dropna().iloc[-1]
                    return float(latest)
    except Exception as e:
        log(f"  国债收益率获取失败: {e}")
    return BOND_YIELD_DEFAULT


def build_index_data(index_code, cfg, start_date, end_date, bond_yield):
    """为单个指数构建数据，多源合并，自动降级"""
    log(f"处理指数: {index_code} {cfg['name']}")
    records = []
    has_price = False

    # 1. 优先从乐咕乐股获取PE/PB/收盘价（上证红利、深证红利）
    legu_df = fetch_legulegu_history(cfg['symbol_legulegu'], start_date, end_date)
    if legu_df is not None and not legu_df.empty:
        records.append(legu_df)
        has_price = True
        log(f"  乐咕乐股数据: {len(legu_df)} 行")

    legu_pb = fetch_legulegu_pb(cfg['symbol_legulegu'], start_date, end_date)
    if legu_pb is not None and not legu_pb.empty:
        records.append(legu_pb)
        log(f"  乐咕乐股PB: {len(legu_pb)} 行")

    # 2. 如果乐咕乐股没有价格，从中证接口获取历史行情（中证红利、红利低波）
    if not has_price:
        cs_hist = fetch_csindex_history(cfg['symbol_csindex'], start_date, end_date)
        if cs_hist is not None and not cs_hist.empty:
            records.append(cs_hist)
            has_price = True
            log(f"  中证历史行情: {len(cs_hist)} 行")

    # 3. 如果仍然没有价格，尝试东方财富（备选）
    if not has_price:
        em_hist = fetch_eastmoney_history(cfg['symbol_eastmoney'], start_date, end_date)
        if em_hist is not None and not em_hist.empty:
            records.append(em_hist)
            has_price = True
            log(f"  东方财富行情: {len(em_hist)} 行")

    if not has_price:
        log(f"  警告: 无行情数据，跳过")
        return pd.DataFrame()

    # 4. 获取PE/股息率（中证接口，所有指数）
    cs_df = fetch_csindex_valuation(cfg['symbol_csindex'], start_date, end_date)
    if cs_df is not None and not cs_df.empty:
        records.append(cs_df)
        log(f"  中证估值: {len(cs_df)} 行")

    # 合并所有数据，处理重叠列（优先取非空值）
    merged = records[0]
    for r in records[1:]:
        overlap_cols = [c for c in r.columns if c in merged.columns and c != 'date']
        if overlap_cols:
            merged = merged.merge(r, on='date', how='outer', suffixes=('_left', '_right'))
            for col in overlap_cols:
                left_col = f"{col}_left"
                right_col = f"{col}_right"
                if left_col in merged.columns and right_col in merged.columns:
                    merged[col] = merged[right_col].fillna(merged[left_col])
                    merged = merged.drop(columns=[left_col, right_col])
        else:
            merged = merged.merge(r, on='date', how='outer')

    merged = merged.sort_values('date')
    merged['index_code'] = index_code
    merged['index_name'] = cfg['name']
    merged['bond_yield'] = bond_yield

    # 选择最终列
    cols = ['date', 'index_code', 'index_name', 'close', 'pe_ttm', 'pe_static', 'pb', 'dividend_yield', 'bond_yield']
    for col in cols:
        if col not in merged.columns:
            merged[col] = None

    return merged[cols].copy()


def update_historical_data(backfill=False):
    """更新历史数据，返回(数据框, 是否有新增数据)"""
    today = datetime.now().strftime('%Y-%m-%d')
    existing_df = load_existing_data()

    all_new_data = []
    bond_yield = fetch_bond_yield()
    log(f"当前10Y国债收益率: {bond_yield}%")

    for code, cfg in INDICES.items():
        if backfill:
            start = cfg['start_date']
        else:
            if not existing_df.empty and code in existing_df['index_code'].values:
                last_date = existing_df[existing_df['index_code'] == code]['date'].max()
                start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                if start > today:
                    log(f"  {code} 数据已是最新，跳过")
                    continue
            else:
                start = cfg['start_date']

        df = build_index_data(code, cfg, start, today, bond_yield)
        if not df.empty:
            all_new_data.append(df)

    if not all_new_data:
        log("没有新数据需要更新")
        return existing_df, False

    new_df = pd.concat(all_new_data, ignore_index=True)

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.sort_values(['index_code', 'date'])
        combined = combined.drop_duplicates(subset=['index_code', 'date'], keep='last')
    else:
        combined = new_df

    combined = combined.sort_values(['index_code', 'date'])
    combined.to_csv(HISTORICAL_CSV, index=False)
    log(f"历史数据已保存: {HISTORICAL_CSV} ({len(combined)} 行)")
    return combined, True


def generate_charts(df):
    """生成图表"""
    if df.empty:
        log("无数据，跳过图表生成")
        return

    CHARTS_DIR.mkdir(exist_ok=True)

    indices = df['index_code'].unique()
    colors = {'000922': '#1f77b4', '000015': '#ff7f0e', '399324': '#2ca02c', 'H30269': '#d62728'}
    axis_dates = df['date']

    # 1. PE trend
    fig, ax = plt.subplots(figsize=(14, 6))
    for code in indices:
        sub = df[(df['index_code'] == code) & (df['pe_ttm'].notna())]
        if not sub.empty:
            ax.plot(
                sub['date'], sub['pe_ttm'],
                label=INDICES[code].get('english_name', INDICES[code]['name']),
                color=colors.get(code, '#333'),
                linewidth=1.5,
            )
    ax.set_title('Dividend Index PE-TTM Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('PE-TTM')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax, axis_dates)
    fig.autofmt_xdate()
    fig.savefig(CHARTS_DIR / 'pe_trend.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"图表已保存: {CHARTS_DIR / 'pe_trend.png'}")

    # 2. PB trend
    fig, ax = plt.subplots(figsize=(14, 6))
    for code in indices:
        sub = df[(df['index_code'] == code) & (df['pb'].notna())]
        if not sub.empty:
            ax.plot(
                sub['date'], sub['pb'],
                label=INDICES[code].get('english_name', INDICES[code]['name']),
                color=colors.get(code, '#333'),
                linewidth=1.5,
            )
    ax.set_title('Dividend Index PB Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('PB')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax, axis_dates)
    fig.autofmt_xdate()
    fig.savefig(CHARTS_DIR / 'pb_trend.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"图表已保存: {CHARTS_DIR / 'pb_trend.png'}")

    # 3. Dividend yield trend
    fig, ax = plt.subplots(figsize=(14, 6))
    for code in indices:
        sub = df[(df['index_code'] == code) & (df['dividend_yield'].notna())]
        if not sub.empty:
            ax.plot(
                sub['date'], sub['dividend_yield'],
                label=INDICES[code].get('english_name', INDICES[code]['name']),
                color=colors.get(code, '#333'),
                linewidth=1.5,
            )
    ax.set_title('Dividend Yield Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Dividend Yield (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax, axis_dates)
    fig.autofmt_xdate()
    fig.savefig(CHARTS_DIR / 'dividend_yield_trend.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"图表已保存: {CHARTS_DIR / 'dividend_yield_trend.png'}")

    # 4. Dividend yield / bond yield spread
    fig, ax = plt.subplots(figsize=(14, 6))
    for code in indices:
        sub = df[(df['index_code'] == code) & (df['dividend_yield'].notna()) & (df['bond_yield'].notna())]
        if not sub.empty:
            ratio = sub['dividend_yield'] / sub['bond_yield']
            ax.plot(
                sub['date'], ratio,
                label=INDICES[code].get('english_name', INDICES[code]['name']),
                color=colors.get(code, '#333'),
                linewidth=1.5,
            )
    ax.axhline(y=2.5, color='green', linestyle='--', alpha=0.5, label='Value Line (2.5x)')
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Low Attractiveness (1.5x)')
    ax.set_title('Dividend Yield / 10Y Bond Yield', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Multiple')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax, axis_dates)
    fig.autofmt_xdate()
    fig.savefig(CHARTS_DIR / 'dy_bond_spread.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"图表已保存: {CHARTS_DIR / 'dy_bond_spread.png'}")

    # 5. Latest valuation summary
    latest_data = []
    for code in indices:
        sub = df[df['index_code'] == code]
        if not sub.empty:
            latest = sub.iloc[-1]
            pe_hist = sub['pe_ttm'].dropna()
            pb_hist = sub['pb'].dropna()
            dy_hist = sub['dividend_yield'].dropna()
            pe_pct = (pe_hist.rank(pct=True).iloc[-1] * 100) if len(pe_hist) > 30 else None
            pb_pct = (pb_hist.rank(pct=True).iloc[-1] * 100) if len(pb_hist) > 30 else None
            dy_pct = ((1 - dy_hist.rank(pct=True).iloc[-1]) * 100) if len(dy_hist) > 30 else None
            latest_data.append({
                'Index': INDICES[code].get('english_name', INDICES[code]['name']),
                'PE Pctl': f"{pe_pct:.1f}%" if pe_pct else 'N/A',
                'PB Pctl': f"{pb_pct:.1f}%" if pb_pct else 'N/A',
                'DY Pctl': f"{dy_pct:.1f}%" if dy_pct else 'N/A',
                'PE': f"{latest['pe_ttm']:.2f}" if pd.notna(latest['pe_ttm']) else 'N/A',
                'PB': f"{latest['pb']:.2f}" if pd.notna(latest['pb']) else 'N/A',
                'DY': f"{latest['dividend_yield']:.2f}%" if pd.notna(latest['dividend_yield']) else 'N/A',
            })

    if latest_data:
        latest_df = pd.DataFrame(latest_data)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        fig.suptitle('Latest Valuation Summary', fontsize=14, fontweight='bold', y=0.98)
        table = ax.table(cellText=latest_df.values, colLabels=latest_df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        fig.savefig(CHARTS_DIR / 'latest_valuation_summary.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        log(f"图表已保存: {CHARTS_DIR / 'latest_valuation_summary.png'}")


def main():
    parser = argparse.ArgumentParser(description='红利指数数据采集')
    parser.add_argument('--backfill', action='store_true', help='回填2020年以来的历史数据')
    parser.add_argument('--charts', action='store_true', help='仅生成图表')
    parser.add_argument('--incremental', action='store_true', help='增量更新（默认）')
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.charts:
        df = load_existing_data()
        generate_charts(df)
    else:
        df, has_new_data = update_historical_data(backfill=args.backfill)
        if args.backfill or has_new_data:
            generate_charts(df)
        else:
            log("没有新增数据，跳过图表生成")

    log("完成")


if __name__ == '__main__':
    main()
