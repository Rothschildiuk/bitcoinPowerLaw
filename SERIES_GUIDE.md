# Series Guide

Read this file only for series behavior, units, filters, mode support, or adding a new series. The canonical source of truth is `core/series_registry.py`.

## Registry Fields
Each series config owns its session keys, defaults, display label/unit, currency behavior, mode availability, chart behavior, optional analysis cutoff, and LogPeriodic parameter bounds.

## Families
- Bitcoin network: Bitcoin, Miner revenue, Bitcoin market cap, Bitcoin volatility, Difficulty, Hashrate.
- Lightning: Lightning nodes, Lightning BTC.
- Liquid: Liquid BTC, Liquid transactions.
- Fiat money: U.S. M2, USDT supply.
- BTC pairs: Filecoin/BTC, Monero/BTC, Litecoin/BTC, Dogecoin/BTC.

## Special Behavior
- Bitcoin is the only series with currency switching (`EUR`, `USD`, `UAH`, `RUB`, `GOLD`, `SILVER`, `COPPER`, `IRON`, `ALUMINUM`) and supports both PowerLaw and LogPeriodic.
- LogPeriodic fits residuals in log time after removing the PowerLaw trend; DSI comparisons use `omega`, `omega,2omega`, and `omega,2omega,4omega`.
- Difficulty and Hashrate support both modes, force log scale, start analysis at `2010-01-01`, and use wider LogPeriodic `Lambda` bounds.
- Bitcoin volatility is PowerLaw-only, log scale, and derived from 30-day daily BTC/USD log-return volatility.
- Bitcoin market cap is PowerLaw-only, log scale, and derived from BTC/USD multiplied by circulating BTC supply.
- Lightning BTC and Liquid BTC display BTC units; Lightning nodes and Liquid transactions display raw units.
- Fiat M2 series are monthly, log scale, no currency conversion, and count PowerLaw time from their own first row.
- BTC-pair series count PowerLaw time from each chain's genesis or first usable launch reference, not Bitcoin genesis.

## Adding a Series
1. Add keys/defaults in `core/constants.py`.
2. Add metadata in `core/series_registry.py`.
3. Add loader/prep in `services/price_service.py`.
4. Wire raw/sidebar data in `app.py`.
5. Add focused tests for registry and loader behavior.

## Maintenance Direction
- Keep series-specific rules in the registry.
- Derive sidebar and chart behavior from the same config.
- Avoid spreading new series routing across hand-written `if/elif` branches.
- Use `make update-defaults` for checked-in PowerLaw and LogPeriodic defaults, then review the `core/constants.py` diff.
