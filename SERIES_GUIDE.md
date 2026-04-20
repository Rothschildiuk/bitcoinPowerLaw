# Series Guide

## Purpose
This file explains how series are organized in the app and where their behavior is defined.

## Canonical Source of Truth
Series metadata lives in `core/series_registry.py`.

Each series config defines:
- session keys for `A` and `B`
- default model parameters
- display name and unit
- currency behavior
- PowerLaw / LogPeriodic availability
- chart behavior such as halving lines
- analysis cutoff for noisy early data
- LogPeriodic parameter bounds overrides where needed

## Series Families

### Bitcoin Network
- Bitcoin
- Miner revenue
- Difficulty
- Hashrate

### Lightning Network
- Lightning nodes
- Lightning BTC

### Liquid
- Liquid BTC
- Liquid transactions

### Fiat Money
- U.S. M2
- Russian M2

## Special Handling

### Bitcoin
- Only series that supports currency switching (`USD`, `EUR`, `GOLD`)
- Available in both PowerLaw and LogPeriodic
- Uses the base LogPeriodic parameter bounds

### Difficulty and Hashrate
- Available in PowerLaw and LogPeriodic
- Force log price scale
- Use startup-era analysis cutoff from `2010-07-18`
- Early raw history may still exist in the cache, but chart/model analysis starts from the cutoff
- Use wider LogPeriodic `Lambda` bounds than Bitcoin
- Their checked-in LogPeriodic defaults are intended to be refreshed from current data

### Lightning BTC and Liquid BTC
- Displayed with `BTC` suffix and 3 decimals

### Liquid Transactions / Lightning Nodes
- Raw unit display, no currency conversion
- Liquid transactions PowerLaw time is counted from the first Liquid transaction row, not from Bitcoin genesis

### U.S. M2
- Uses FRED `M2SL`
- Displayed in billions of U.S. dollars
- Monthly data, no currency conversion
- Forces log price scale
- PowerLaw time is counted from the U.S. M2 row start, not from Bitcoin genesis
- M2 is used instead of M3/M4 because it is the current official broad Fed/FRED money stock series.

### Russian M2
- Uses Bank of Russia monetary aggregates as the primary source
- Falls back to FRED `MYAGM2RUM189N` if the CBR workbook is unavailable
- Displayed in trillions of Russian rubles
- Monthly data, no currency conversion
- Forces log price scale
- PowerLaw time is counted from the Russian M2 row start, not from Bitcoin genesis
- M2 is used because it is the broad official local-currency aggregate with a current CBR-published history.

## When Adding a New Series
1. Add defaults and keys in `core/constants.py`
2. Add metadata in `core/series_registry.py`
3. Add loader in `services/price_service.py`
4. Register raw/sidebar data in `app.py`
5. Add tests for registry and loader behavior

## Default Maintenance
- Checked-in defaults are not purely hand-tuned anymore.
- Use `make update-defaults` to refresh:
  - PowerLaw `A/B`
  - LogPeriodic defaults for supported series
- Review the resulting `core/constants.py` diff before committing.

## Preferred Refactor Direction
- Put series-specific rules in the registry
- Keep sidebar and chart behavior derived from the same config
- Avoid reintroducing hand-written `if/elif` routing across multiple files
