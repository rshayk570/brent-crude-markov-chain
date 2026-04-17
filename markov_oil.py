import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load/clean data
df = pd.read_csv('archive/oil_prices_daily.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()
df = df[df['brent_30d_vol'] > 0]       # drop early lag period (first ~40 rows)
df = df[df.index < '2025-04-04']        # drop placeholder/constant-value tail

# define volatility regime states
low_thresh  = df['brent_30d_vol'].quantile(0.33)
high_thresh = df['brent_30d_vol'].quantile(0.67)

def classify_regime(v):
    if v < low_thresh:
        return 'LOW'
    elif v < high_thresh:
        return 'MED'
    return 'HIGH'

df['regime'] = df['brent_30d_vol'].apply(classify_regime)

print(f"Volatility thresholds:  LOW < {low_thresh:.2f}  |  MED < {high_thresh:.2f}  |  HIGH >= {high_thresh:.2f}")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}  ({len(df):,} trading days)\n")