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

## Special Handling

### Bitcoin
- Only series that supports currency switching (`USD`, `EUR`, `GOLD`)
- Available in both PowerLaw and LogPeriodic

### Difficulty and Hashrate
- Available in PowerLaw and LogPeriodic
- Force log price scale
- Use startup-era analysis cutoff from `2010-07-18`
- Early raw history may still exist in the cache, but chart/model analysis starts from the cutoff

### Lightning BTC and Liquid BTC
- Displayed with `BTC` suffix and 3 decimals

### Liquid Transactions / Lightning Nodes
- Raw unit display, no currency conversion

## When Adding a New Series
1. Add defaults and keys in `core/constants.py`
2. Add metadata in `core/series_registry.py`
3. Add loader in `services/price_service.py`
4. Register raw/sidebar data in `app.py`
5. Add tests for registry and loader behavior

## Preferred Refactor Direction
- Put series-specific rules in the registry
- Keep sidebar and chart behavior derived from the same config
- Avoid reintroducing hand-written `if/elif` routing across multiple files
