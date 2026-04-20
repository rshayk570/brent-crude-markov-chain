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

# steady state distribution
eigenvalues, eigenvectors = np.linalg.eig(trans_matrix.values.T)
dominant_idx = np.argmax(np.abs(eigenvalues))
stationary = np.real(eigenvectors[:, dominant_idx])
stationary = stationary / stationary.sum()
print("── Stationary Distribution (long-run regime probabilities) ──────────────")
for s, p in zip(states, stationary):
    print(f"  {s}: {p:.4f} ({p*100:.1f}%)")
print()


# regime duration statistics
df['regime_run'] = (df['regime'] != df['regime'].shift()).cumsum()
durations = (df.groupby(['regime_run', 'regime'])
               .size()
               .reset_index(name='days')
               .groupby('regime')['days'])
dur_stats = durations.agg(['count', 'mean', 'median', 'std', 'min', 'max']).T
dur_stats.columns.name = 'regime'
print("── Regime Persistence (consecutive days per stint) ──────────────────────")
for regime in states:
    s = durations.get_group(regime)
    print(f"  {regime:4s}  stints={int(s.count()):4d}  "
          f"mean={s.mean():.1f}d  median={s.median():.0f}d  "
          f"std={s.std():.1f}d  max={int(s.max()):d}d")
print()

# regime frequency
print("── Observed State Frequency ─────────────────────────────────────────────")
print(df['regime'].value_counts(normalize=True).round(4).to_string())
print()

# markov transition heatmap (plot #1)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    trans_matrix, annot=True, fmt='.3f', cmap='Blues',
    linewidths=0.5, vmin=0, vmax=1,
    xticklabels=states, yticklabels=states, ax=ax
)
ax.set_title('Markov Transition Matrix\nBrent Crude Volatility Regimes (1990–2025)')
ax.set_xlabel('To State')
ax.set_ylabel('From State')
plt.tight_layout()
plt.savefig('transition_heatmap.png', dpi=150)
print("Saved: transition_heatmap.png")

# plot 2 - regime/price timeline
colors = {'LOW': '#2ca02c', 'MED': '#ff7f0e', 'HIGH': '#d62728'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

for regime in states:
    grp = df[df['regime'] == regime]
    ax1.scatter(grp.index, grp['brent_usd'],
                c=colors[regime], s=1.5, label=regime, alpha=0.8, zorder=2)

# geopolitical events
geo = df[df['is_geopolitical_event'] == 1]
if not geo.empty:
    ax1.scatter(geo.index, geo['brent_usd'],
                marker='^', c='black', s=20, zorder=3, label='Geo event', alpha=0.5)

ax1.set_ylabel('Brent Crude (USD)')
ax1.set_title('Brent Price Colored by Volatility Regime (1990–2025)')
ax1.legend(markerscale=4, loc='upper left')
ax1.set_ylim(0, 250)
ax1.grid(axis='y', alpha=0.3)

ax2.plot(df.index, df['brent_30d_vol'], lw=0.7, color='steelblue', label='30d Vol')
ax2.axhline(low_thresh,  color=colors['LOW'],  linestyle='--', lw=1.2,
            label=f'Low/Med ({low_thresh:.1f})')
ax2.axhline(high_thresh, color=colors['HIGH'], linestyle='--', lw=1.2,
            label=f'Med/High ({high_thresh:.1f})')
ax2.fill_between(df.index, df['brent_30d_vol'], alpha=0.15, color='steelblue')
ax2.set_ylim(0, df['brent_30d_vol'].max() * 1.12)
ax2.set_ylabel('30-Day Volatility (%)')
ax2.set_xlabel('Date')
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('regime_timeline.png', dpi=150)
print("Saved: regime_timeline.png")

plt.show()

# live data and current prediction
EIA_API_KEY = os.environ.get('EIA_API_KEY', '')
if not EIA_API_KEY:
    raise EnvironmentError("Set EIA_API_KEY environment variable before running.")

print("── Live EIA Forecast ────────────────────────────────────────────────────")

resp = requests.get(
    'https://api.eia.gov/v2/petroleum/pri/spt/data/',
    params={
        'api_key':              EIA_API_KEY,
        'frequency':            'daily',
        'data[0]':              'value',
        'facets[product][]':    'EPCBRENT',
        'sort[0][column]':      'period',
        'sort[0][direction]':   'desc',
        'length':               60,
    },
    timeout=15,
)
resp.raise_for_status()

records = resp.json()['response']['data']
live = (pd.DataFrame(records)[['period', 'value']]
          .rename(columns={'period': 'date', 'value': 'brent_usd'})
          .assign(date=lambda d: pd.to_datetime(d['date']))
          .sort_values('date')
          .set_index('date'))
live['brent_usd'] = pd.to_numeric(live['brent_usd'])

# replicate brent_30d_vol: rolling 30-day std of daily % return
live['daily_return_pct'] = live['brent_usd'].pct_change() * 100
live['vol_30d'] = live['daily_return_pct'].rolling(30).std()
live = live.dropna(subset=['vol_30d'])

latest       = live.index[-1]
current_vol  = live['vol_30d'].iloc[-1]
current_price = live['brent_usd'].iloc[-1]
current_regime = classify_regime(current_vol)

# days already spent in this regime (current streak length)
streak = 1
for v in live['vol_30d'].iloc[-2::-1]:
    if classify_regime(v) == current_regime:
        streak += 1
    else:
        break

# geometric distribution: expected total stay = 1 / (1 - p_ii)
p_self = trans_matrix.loc[current_regime, current_regime]
expected_total = 1 / (1 - p_self)
# Memoryless property: expected remaining = same regardless of streak
expected_remaining = expected_total

print(f"  Latest EIA date : {latest.date()}")
print(f"  Brent price     : ${current_price:.2f}/bbl")
print(f"  30d volatility  : {current_vol:.3f}  (thresholds: LOW<{low_thresh:.2f}, HIGH≥{high_thresh:.2f})")
print(f"  Current regime  : {current_regime}")
print(f"  Streak so far   : {streak} trading days in {current_regime}")
print(f"  Self-transition : {p_self:.4f}")
print(f"  Expected remaining duration : {expected_remaining:.1f} trading days")
print(f"  (geometric mean; Markov property makes this independent of streak length)")
