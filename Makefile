VENV := venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
STREAMLIT := $(PY) -m streamlit
BLACK := $(PY) -m black
BLACK_CONFIG := --config config/pyproject.toml
BLACK_TARGETS := app.py core services tests ui

.PHONY: help install run format check test clean update-defaults update-data-snapshots

$(PY):
	python3 -m venv $(VENV)

help:
	@echo "Targets:"
	@echo "  make install  - install/update python dependencies into venv"
	@echo "  make run      - run Streamlit app"
	@echo "  make format   - run black formatter"
	@echo "  make check    - run black --check"
	@echo "  make test     - run pytest test suite"
	@echo "  make update-defaults - recompute and rewrite PowerLaw and LogPeriodic defaults"
	@echo "  make update-data-snapshots - refresh checked-in daily snapshot CSV files"
	@echo "  make clean    - remove Python cache folders"

install: $(PY)
	$(PIP) install -U pip
	$(PIP) install -r config/requirements.txt

run: $(PY)
	$(STREAMLIT) run app.py

format: $(PY)
	$(BLACK) $(BLACK_CONFIG) $(BLACK_TARGETS)

check: $(PY)
	$(BLACK) $(BLACK_CONFIG) --check $(BLACK_TARGETS)

test: $(PY)
	$(PY) -m pytest -q

update-defaults: $(PY)
	$(PY) scripts/update_powerlaw_defaults.py

update-data-snapshots: $(PY)
	$(PY) scripts/update_data_snapshots.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
