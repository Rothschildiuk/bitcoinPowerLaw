# Architecture

## Purpose
Streamlit app for exploring Bitcoin-style PowerLaw models across price, network, liquidity, fiat, and ecosystem series.

## Runtime Flow
1. `app.py` loads prepared datasets via `services/price_service.py`.
2. Defaults and series metadata come from `core/constants.py` and `core/series_registry.py`.
3. `ui/sidebar.py` resolves mode, series, currency, filters, and parameters.
4. `app.py` builds the active display frame and calls model helpers.
5. `ui/charts.py`, `ui/kpi.py`, and `ui/theme.py` render the UI.

## Key Modules
- `app.py`: main entrypoint, orchestration, portfolio view, shared data prep
- `core/constants.py`: app constants, defaults, labels, session keys
- `core/series_registry.py`: canonical series metadata and grouping
- `core/power_law.py`: PowerLaw regression, bounds, and controls
- `services/price_service.py`: snapshot-first data loaders, normalization, and maintenance cache
- `ui/sidebar.py`: mode/series/parameter selection
- `ui/charts.py`: Plotly chart assembly and axis helpers
- `ui/kpi.py`: fair value and history-share KPI cards
- `ui/viewport.py`: User-Agent based phone detection for server-side layout choices
- `scripts/update_powerlaw_defaults.py`: maintenance script for checked-in PowerLaw defaults

## Design Rules
- Keep raw external data loading in `services/price_service.py`.
- Prefer checked-in snapshots at runtime; use maintenance scripts for network refreshes.
- Keep series-specific behavior in `core/series_registry.py`.
- Prefer registry wiring over new `if/elif` routing in `app.py`.
- Keep display filtering and analysis filtering aligned between sidebar and main chart.
- Responsive layout is split by what can still change in the browser: sizes, spacing and
  visibility belong in `assets/styles.css` media queries, while anything Plotly bakes into
  the figure (height, margins, tick density, modebar buttons) is chosen in Python from
  `ui.viewport.is_mobile_client`.
- When defaults are meant to track current datasets, update them through `scripts/update_powerlaw_defaults.py` instead of editing `core/constants.py` by hand.

## Current Important Constraints
- PowerLaw chart horizon extends 10 years beyond the newer of today and the latest data point.
- Difficulty and Hashrate use an analysis cutoff starting at `2010-01-01` to skip the earliest startup-era rows.
- Currency conversion applies only to Bitcoin price series.
- The portfolio strategy tester is walk-forward: the withdrawal floor is refitted for
  each backtested month from data available up to that month
  (`core/power_law.build_causal_powerlaw_floor_prices`). Never size a backtest from a
  model fitted on the full history.
