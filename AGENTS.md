# AGENTS.md

## Workflows
- Setup: `python3 -m venv venv`, `make install`, `npm install`.
- Discover commands: `make help`.
- Run app: `make run`; macOS shortcut: `./run.command`.
- Format/check/test: `make format`, `make check`, `make test`.
- JS tests: `npm test` is a placeholder and exits 0 with "No JS tests configured".
- Refresh defaults: `make update-defaults`; preview with `venv/bin/python scripts/update_powerlaw_defaults.py --dry-run`.
- Cleanup: `make clean`.

## Project Notes
- Streamlit app. Python deps come from `config/requirements.txt`; transitive pins live in `config/constraints.txt`.
- Root `requirements.txt` delegates to `config/requirements.txt`; keep the constraints reference relative to `config/` for Streamlit Cloud.
- `run.command` runs `make install` when `venv/bin/streamlit` is missing.
- `make update-defaults` rewrites PowerLaw `A/B` and LogPeriodic defaults in `core/constants.py`.
- `scripts/playwright/` exists, but no committed entry command is present.

## Context Budget
- Start with `ARCHITECTURE.md` for module flow.
- Read `DATA_SOURCES.md` only for source/cache/refresh work.
- Read `SERIES_GUIDE.md` only for series behavior, units, or new-series work.
- Avoid searching `data/snapshots/**` unless the task is specifically about snapshot contents; those CSVs dominate token output.

## Test Budget
- Do not run tests after every small change.
- For low-risk edits, rely on code inspection and minimal diff review.
- For risky changes, mention the targeted test command first; run it only when requested or when the app is clearly broken.
