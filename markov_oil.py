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

# build transition count & probability matrix
states = ['LOW', 'MED', 'HIGH']

from_states = df['regime'].iloc[:-1].values
to_states   = df['regime'].iloc[1:].values

counts = pd.DataFrame(0, index=states, columns=states, dtype=int)
for f, t in zip(from_states, to_states):
    counts.loc[f, t] += 1

trans_matrix = counts.div(counts.sum(axis=1), axis=0)

print("── Transition Count Matrix ──────────────────────────────────────────────")
print(counts.to_string())
print()
print("── Transition Probability Matrix ────────────────────────────────────────")
print(trans_matrix.round(4).to_string())
print()

# verify rows sum to 1
assert np.allclose(trans_matrix.sum(axis=1), 1.0), "Row sums should equal 1"

# ── Steady-State Distribution ─────────────────────────
eigenvalues, eigenvectors = np.linalg.eig(trans_matrix.values.T)
dominant_idx = np.argmax(np.abs(eigenvalues))
stationary = np.real(eigenvectors[:, dominant_idx])
stationary = stationary / stationary.sum()
print("── Stationary Distribution (long-run regime probabilities) ──────────────")
for s, p in zip(states, stationary):
    print(f"  {s}: {p:.4f} ({p*100:.1f}%)")
print()
