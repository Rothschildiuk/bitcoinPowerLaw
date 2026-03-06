# Data Sources

## Overview
Prepared datasets are loaded through `services/price_service.py`. The module normalizes upstream payloads into frames with:
- `Close`
- `AbsDays`
- `LogClose`

Most prepared datasets are cached on disk under `output/data_cache/`.

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

### Miner Revenue / Difficulty / Hashrate
- Source: Blockchain.com CSV chart endpoints
- These endpoints behave like headerless CSV payloads and must be parsed with explicit column names.

### Lightning
- Source: `bitcoinvisuals.com/static/data/data_daily.csv`
- Derived prepared series:
  - Lightning nodes
  - Lightning capacity

### Liquid
- Official reserves:
  - `https://liquid.network/api/v1/liquid/reserves`
  - `https://liquid.network/api/v1/liquid/reserves/month`
- Site chart data:
  - `https://liquid.net/api/getChartsData`
- Derived prepared series:
  - Liquid BTC
  - Liquid transactions

## Cache Model
- Disk cache dir: `output/data_cache/`
- Cache metadata is stored as `.meta.json` alongside cached CSV files.
- Cache schema version is used to invalidate stale serialization/layout assumptions.
- If refresh fails and a valid cached snapshot exists, the app serves cached data.

## Refresh Intervals
- Fast series: 1 hour
- Slow series: 6 hours
- Reference series: 12 hours

## Reliability Notes
- Blockchain.com early difficulty/hashrate rows contain noisy startup-era data.
- Difficulty/hashrate analysis is intentionally filtered from `2010-07-18` onward in the app layer.
- Some upstreams are unofficial or undocumented and should be treated as operational dependencies, not strict contracts.
