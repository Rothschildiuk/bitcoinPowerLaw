# Architecture

## Purpose
This project is a Streamlit app for exploring Bitcoin-style power law and log-periodic models across multiple on-chain and ecosystem series.

## Runtime Flow
1. `app.py` loads prepared datasets from `services/price_service.py`.
2. Session defaults are initialized from `core/constants.py` and `core/series_registry.py`.
3. `ui/sidebar.py` resolves mode, active series, currency, and model parameters.
4. `app.py` builds the filtered display frame for the active series.
5. Modeling logic runs through:
   - `core/power_law.py` for power regression and parameter controls
   - `core/oscillator.py` for log-periodic residual fitting
   - `scripts/update_powerlaw_defaults.py` for recomputing checked-in model defaults
6. Rendering is handled by:
   - `ui/charts.py` for the main chart
   - `ui/kpi.py` for KPI cards
   - `ui/theme.py` for colors and styling

## Key Modules
- `app.py`: main entrypoint, orchestration, portfolio view, shared data prep
- `core/constants.py`: app constants, defaults, labels, session keys
- `core/series_registry.py`: canonical series metadata and grouping
- `core/power_law.py`: regression helpers and PowerLaw sidebar controls
- `core/oscillator.py`: oscillator fitting, LogPeriodic sidebar controls, and auto-fit logic
- `services/price_service.py`: snapshot-first data loaders, normalization, and maintenance cache
- `ui/sidebar.py`: mode/series/parameter selection
- `ui/charts.py`: Plotly chart assembly and axis helpers
- `ui/kpi.py`: fair value and history-share KPI cards
- `scripts/update_powerlaw_defaults.py`: maintenance script for checked-in PowerLaw and LogPeriodic defaults

## Design Rules
- Keep raw external data loading inside `services/price_service.py`.
- Prefer checked-in snapshot datasets for runtime reads; use network fetches through maintenance scripts instead of per-page-load refreshes.
- Keep series-specific behavior inside `core/series_registry.py`.
- Prefer wiring from registry instead of adding new `if/elif` branches in `app.py`.
- Keep display filtering and analysis filtering aligned between sidebar and main chart.
- When defaults are meant to track current datasets, update them through `scripts/update_powerlaw_defaults.py` instead of editing `core/constants.py` by hand.

## Current Important Constraints
- PowerLaw chart horizon is limited to 5 years beyond the latest data point.
- Difficulty and Hashrate use an analysis cutoff starting at `2010-07-18` to avoid early network noise.
- Currency conversion applies only to Bitcoin price series.
- LogPeriodic sidebar currently exposes only `1st Cycle Age` and `Lambda`; amplitude and damping stay pinned to series defaults.
- Difficulty and Hashrate allow a wider LogPeriodic `Lambda` search range than Bitcoin.
