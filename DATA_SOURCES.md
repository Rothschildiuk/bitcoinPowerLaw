# Data Sources

Read this file only when changing loaders, cache behavior, snapshot refreshes, or source-specific reliability logic. Runtime loaders live in `services/price_service.py`; checked-in snapshots live in `data/snapshots/`.

## Data Model
- Prepared frames expose `Close`, `AbsDays`, and `LogClose`.
- Runtime prefers checked-in snapshots to avoid page-load network fetches.
- Runtime cache lives under ignored `output/data_cache/` with adjacent `.meta.json` files and schema-version invalidation.
- Avoid broad searches in `data/snapshots/**`; inspect specific files only when snapshot contents matter.

## Sources
- Bitcoin price: GitHub CSV mirror; tail fallback order is `yfinance` `BTC-USD`, CoinGecko range API, then CoinCap history API.
- Bitcoin volatility: derived from BTC/USD as 30-day rolling std of daily log returns times 100.
- Bitcoin market cap: BTC/USD multiplied by Blockchain.com `total-bitcoins` circulating supply.
- FX/metals/index references: `yfinance` symbols `EURUSD=X`, `UAH=X`, `RUB=X`, `GC=F`, `SI=F`, `HG=F`, `TIO=F`, `ALI=F`, `CL=F`, `^GSPC`, `^IXIC`, fallback `XAUUSD=X`.
- Filecoin/Monero/Litecoin/Dogecoin BTC pairs: CryptoCompare USD history converted through BTC/USD.
- Miner revenue, Difficulty, Hashrate, Bitcoin circulating supply: Blockchain.com headerless CSV chart endpoints.
- Lightning nodes/capacity: `bitcoinvisuals.com/static/data/data_daily.csv`.
- Liquid BTC/transactions: Liquid reserves API plus `liquid.net/api/getChartsData`.
- U.S. M2: FRED `M2SL`, billions USD, monthly.
- USDT supply: DeFiLlama stablecoin API, billions of circulating pegged USD for Tether, daily.
- COFER reserve currency dominance: IMF COFER SDMX 3.0 `AFXRA` + `SHRO_PT`, percent share of allocated official FX reserves by currency, quarterly. BTC is derived as Bitcoin market capitalization divided by IMF COFER total world official FX reserves `TFXRA` + `NV_USD`.

## Origins and Filters
- Filecoin origin: official mainnet genesis reset timestamp `2020-08-24T22:00:00Z`.
- Monero origin: first mined mainnet block date `2014-04-18` because block 0 timestamp is `0`.
- Litecoin origin: block 0 timestamp `2011-10-07`.
- Dogecoin origin: block 0 timestamp `2013-12-06`.
- USDT supply origin: first DeFiLlama stablecoin API row `2017-11-29`.
- COFER reserve currency dominance origin: first quarterly rows `1999-03-31`; newer currencies start when IMF begins identifying them separately.
- Lightning origin: first checked-in snapshot row `2018-01-19`.
- Liquid BTC origin: first checked-in row `2018-09-01`; Liquid transactions origin: `2018-09-24`.
- Difficulty/Hashrate raw early rows can be noisy; app analysis starts at `2010-01-01`.

## Maintenance
- App runtime uses checked-in snapshots by default (`POWERLAW_DATA_SOURCE=snapshot`) so page loads do not wait on external APIs. For local debugging, set `POWERLAW_DATA_SOURCE=auto` or `POWERLAW_DATA_SOURCE=live`.
- Refresh snapshots: `make update-data-snapshots` or `venv/bin/python scripts/update_data_snapshots.py`.
- Refresh model defaults: `make update-defaults`.
- Refresh both in the correct order: `make update-all-data`.
- GitHub Actions runs `.github/workflows/refresh-data.yml` at `00:00`, `06:00`, `12:00`, and `18:00` Europe/Vienna to refresh snapshots and recompute checked-in constants. It can also be started manually with `workflow_dispatch`.
- Preview defaults: `venv/bin/python scripts/update_powerlaw_defaults.py --dry-run`.
- Default refresh rewrites PowerLaw `A/B` and LogPeriodic defaults in `core/constants.py`.
