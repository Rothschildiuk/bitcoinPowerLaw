# Data Sources

## Overview
Prepared datasets are loaded through `services/price_service.py`. The module normalizes upstream payloads into frames with:
- `Close`
- `AbsDays`
- `LogClose`

Most prepared datasets are cached on disk under `output/data_cache/`.
Checked-in runtime snapshots are stored under `data/snapshots/`.

## Main Sources

### Bitcoin Price
- Primary history: GitHub CSV mirror
- Tail refresh fallback order:
  1. `yfinance` (`BTC-USD`)
  2. CoinGecko range API
  3. CoinCap history API

### FX and Gold Reference Series
- `yfinance`
- Symbols:
  - `EURUSD=X`
  - `GC=F`
  - fallback `XAUUSD=X`

### Shitcoins BTC Pairs
- History source: CryptoCompare daily history in `USD`, converted to `BTC` via `BTC/USD`
- Covered series:
  - Filecoin / BTC
  - Monero / BTC
  - Litecoin / BTC
  - Dogecoin / BTC

### Miner Revenue / Difficulty / Hashrate
- Source: Blockchain.com CSV chart endpoints
- These endpoints behave like headerless CSV payloads and must be parsed with explicit column names.

### Lightning
- Source: `bitcoinvisuals.com/static/data/data_daily.csv`
- Derived prepared series:
  - Lightning nodes
  - Lightning capacity
- Model origin: first checked-in Lightning snapshot row (`2018-01-19`), consistent with early 2018 mainnet usage and before the first `lnd` mainnet beta release on `2018-03-15`.

### Liquid
- Official reserves:
  - `https://liquid.network/api/v1/liquid/reserves`
  - `https://liquid.network/api/v1/liquid/reserves/month`
- Site chart data:
  - `https://liquid.net/api/getChartsData`
- Derived prepared series:
  - Liquid BTC
  - Liquid transactions
- Network reference: Blockstream says the Liquid blockchain went live on `2018-09-27`.
- Model origins follow the first checked-in rows instead of the later launch reference when the source provides earlier aggregate rows: Liquid BTC starts at `2018-09-01`; Liquid transactions starts at `2018-09-24`.

### U.S. M2 Money Supply
- Source: FRED CSV endpoint for `M2SL`
- URL: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL`
- Unit: billions of U.S. dollars
- Frequency: monthly
- The app uses M2 because it is the broadest current official Federal Reserve money stock aggregate published through H.6/FRED. M3 was discontinued by the Federal Reserve in 2006, and broader M4/Divisia aggregates require third-party methodology.

### Russian M2 Money Supply
- Primary source: Bank of Russia monetary aggregates workbook
- URL: `https://www.cbr.ru/vfs/eng/statistics/credit_statistics/monetary_agg_e.xlsx`
- Fallback source: FRED CSV endpoint for `MYAGM2RUM189N`
- Fallback URL: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=MYAGM2RUM189N`
- Unit: trillions of Russian rubles
- Frequency: monthly
- The app uses national M2 because it is the broad local-currency aggregate with the longest official current row available from CBR. The FRED row is shorter and is kept only as a resilience fallback.

## Cache Model
- Disk cache dir: `output/data_cache/`
- Cache metadata is stored as `.meta.json` alongside cached CSV files.
- Cache schema version is used to invalidate stale serialization/layout assumptions.
- If refresh fails and a valid cached snapshot exists, the app serves cached data.
- Cache files are local runtime artifacts and are intentionally ignored by git.

## Snapshot Model
- Runtime-preferred datasets live in `data/snapshots/`.
- The Streamlit app reads these checked-in snapshots first to avoid external fetches on page load.
- Refresh workflow:
  - `make update-data-snapshots`
  - `venv/bin/python scripts/update_data_snapshots.py`
- Snapshot refresh is an explicit maintenance step and can be run periodically instead of per-request.

## Refresh Intervals
- Fast series: 1 hour
- Slow series: 6 hours
- Reference series: 12 hours

## Reliability Notes
- Blockchain.com early difficulty/hashrate rows contain noisy startup-era data.
- Difficulty/hashrate analysis is intentionally filtered from `2010-07-18` onward in the app layer.
- Some upstreams are unofficial or undocumented and should be treated as operational dependencies, not strict contracts.

## Default Refresh Workflow
- `make update-defaults` recomputes checked-in defaults from the current cached/live datasets.
- The script updates:
  - PowerLaw `A/B` defaults
  - LogPeriodic defaults for Bitcoin, Difficulty, and Hashrate
- Safe preview command:
  - `venv/bin/python scripts/update_powerlaw_defaults.py --dry-run`
