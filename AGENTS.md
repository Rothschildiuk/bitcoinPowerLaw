# AGENTS.md

## Scope
- Python Streamlit app workflow (run, format, test).
- Node helper workflow for Gemini CLI prompts.

## Verified Workflows
- Initial setup (fresh clone):
  - `python3 -m venv venv`
  - `make install`
- Run app:
  - `make run`
  - macOS shortcut: `./run.command`
- Format and style checks:
  - `make format`
  - `make check`
- Tests:
  - `make test`
- Cleanup:
  - `make clean`
- Gemini CLI helpers:
  - `npm run gemini`
  - `npm run gemini:review`

## Grounded Notes
- Python dependencies are sourced from `config/requirements.txt`.
- `requirements.txt` points to `config/requirements.txt` via `-r`.
- `npm test` is a placeholder and currently does not run JS tests.

## TODO
- `scripts/playwright/` exists, but no committed entry command is present yet.
