# Learned Repo Memory

## Memory Metadata
- created_at_unix: 1773364273
- created_at_utc: 2026-03-13T01:11:13Z
- refreshed_at_unix: 1774099785
- refreshed_at_utc: 2026-03-21T13:29:45Z
- refresh_policy: Refresh this file if it is older than 7 days and repo development has continued.

## Repo
- name: PowerLaw
- purpose: Streamlit app for exploring Bitcoin-style PowerLaw and LogPeriodic models across Bitcoin, on-chain, and ecosystem series.
- project types: node, python
- primary languages: python, javascript

## Read First
- `AGENTS.md`: agent instructions
- `.context-pack/memory.md`: learned repo memory
- `ARCHITECTURE.md`: architecture guide
- `DATA_SOURCES.md`: data source guide
- `SERIES_GUIDE.md`: domain guide

## Entry Points
- `app.py`: entrypoint-like source file, language-aware boost (python, top-1)
- `Makefile`: build or orchestration entrypoint, active work

## Stack
- Python Streamlit app with Plotly-based chart rendering.
- Core model code lives under `core/`; runtime data loading and normalization live under `services/`.
- Node tooling is auxiliary, currently for Gemini CLI helpers and an in-progress browser smoke workflow.
- Python dependencies are pinned via `config/requirements.txt` and `config/constraints.txt`.

## Local Workflows
- Fresh setup: `python3 -m venv venv`, `make install`, `npm install`.
- Run UI locally: `make run` or `./run.command`.
- Python quality gates: `make format`, `make check`, `make test`.
- Data/default maintenance: `make update-data-snapshots`, `make update-defaults`.
- Node helpers: `npm run gemini`, `npm run gemini:review`.

## Service Topology
- Single-process Streamlit UI in `app.py`; there is no separate backend service.
- `services/price_service.py` is the integration boundary for external data and cache/snapshot loading.
- Runtime prefers checked-in snapshot CSVs from `data/snapshots/`; on-disk cache artifacts live under `output/data_cache/`.
- UI state and model parameter selection flow through `ui/sidebar.py`, while rendering is split between `ui/charts.py`, `ui/kpi.py`, and `ui/theme.py`.

## Hotspots
- `Makefile`: build or orchestration entrypoint, active work
- `app.py`: entrypoint-like source file, language-aware boost (python, top-1)
- `core/series_registry.py`: large production source file
- `services/price_service.py`: large production source file
- `package.json`: project manifest, active work

## Known Pitfalls
- No README found.
- tree entries omitted by limit: 15
- `config/requirements.txt` must reference `constraints.txt` relative to `config/` or Streamlit Cloud installs break.
- Defaults in `core/constants.py` should be refreshed by `scripts/update_powerlaw_defaults.py`, not edited by hand.
- Difficulty and Hashrate intentionally cut off analysis before `2010-07-18`; raw cached history may still be older.
- `package-lock.json` is tracked, so any committed `package.json` dependency change should usually be accompanied by a lockfile refresh.

## Operational Notes
- `config/requirements.txt`: dependencies `-c constraints.txt`, `streamlit`, `pandas`, `numpy`.
- `run.command` bootstraps dependencies automatically if `venv/bin/streamlit` is missing.
- There is no committed CI workflow under `.github/workflows/`; repository automation currently relies on local `make` and script workflows.

## Debugging Notes
- Portfolio mode now supports an experimental monthly-buy scenario with separate portfolio, invested-cash, and BTC-balance projections.
- `make run` and `run.command` intentionally use `python -m ...` entrypoints so moved repos still work even if old venv wrapper shebangs are stale.

## Open Questions
- If `agent-browser` stays as a tracked devDependency, the repo should decide whether `package-lock.json` must always be kept in sync before merge.
