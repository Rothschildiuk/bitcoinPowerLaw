# AGENTS.md

## Scope
- Python Streamlit app workflow (run, format, test).
- Node helper workflow for Gemini CLI prompts.

## Verified Workflows
- Initial setup (fresh clone):
  - `python3 -m venv venv`
  - `make install`
  - `npm install`
- Command discovery:
  - `make help`
- Run app:
  - `make run`
  - macOS shortcut: `./run.command`
- Format and style checks:
  - `make format`
  - `make check`
- Tests:
  - `make test`
  - `npm test` (placeholder; exits 0 with "No JS tests configured")
- Cleanup:
  - `make clean`
- Gemini CLI helpers:
  - `npm run gemini`
  - `npm run gemini:review`

## Grounded Notes
- Python dependencies are sourced from `config/requirements.txt`.
- Transitive Python versions are pinned via `config/constraints.txt` for reproducible installs.
- `requirements.txt` points to `config/requirements.txt` via `-r`.
- `npm test` is a placeholder and currently does not run JS tests.
- `run.command` runs `make install` automatically when `venv/bin/streamlit` is missing.

## Fast Context Files
- `ARCHITECTURE.md` explains the runtime flow and main modules.
- `DATA_SOURCES.md` lists external APIs, cache behavior, and reliability notes.
- `SERIES_GUIDE.md` explains series families, units, filters, and mode support.

## TODO
- `scripts/playwright/` exists, but no committed entry command is present yet.
