Markov Chain Volatility Regime Analysis — Brent Crude Oil
==========================================================

REQUIREMENTS
------------
Python 3.x
Get a free EIA API key — register at: https://www.eia.gov/opendata/ to compute current data. 

SETUP
-----
1. Clone or download this directory.

2. Create the virtual environment and install dependencies:

       python3 -m venv prob-stat-venv
       prob-stat-venv/bin/pip install pandas numpy matplotlib seaborn requests

3. Set your EIA API key as an environment variable:

       export EIA_API_KEY=your_key_here

DATA
----
Ensure the source CSV is in archive directory:

    archive/oil_prices_daily.csv

The following columns are needed for computation: date, brent_usd,
brent_daily_return_pct, brent_30d_vol, is_geopolitical_event.

EXECUTION
---------
From the project directory, run:

    prob-stat-venv/bin/python markov_oil.py


OUTPUT
------
Console output:
  - Volatility regime thresholds and date range
  - Transition count and probability matrices
  - Stationary distribution
  - Regime persistence statistics (mean/median/std/max days per stint)
  - Observed state frequencies
  - Live EIA forecast: current regime, streak, and expected remaining duration

Image files (saved to project root):
  - transition_heatmap.png   — 3x3 Markov transition probability heatmap
  - regime_timeline.png      — 35-year price and volatility chart colored by regime

NOTES
-----
- Rows with brent_30d_vol == 0 are dropped. 
- Rows after 2025-04-03 are dropped
- The EIA live fetch retrieves the 60 most recent daily Brent spot prices to
  recompute current 30d rolling volatility
