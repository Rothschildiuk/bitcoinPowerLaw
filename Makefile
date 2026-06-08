VENV := venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
STREAMLIT := $(PY) -m streamlit
BLACK := $(PY) -m black
BLACK_CONFIG := --config config/pyproject.toml
BLACK_TARGETS := app.py core services tests ui
PY_TARGETS := app.py core services tests ui
UI_TEST_TARGETS := tests/test_oscillator.py tests/test_ui_charts_helpers.py tests/test_ui_sidebar_helpers.py
UI_CHECK_TARGETS := core/oscillator.py ui/charts.py ui/sidebar.py tests/test_oscillator.py tests/test_ui_charts_helpers.py tests/test_ui_sidebar_helpers.py

.PHONY: help install run format format-ui check check-ui compile test test-ui test-charts test-oscillator test-sidebar verify clean update-defaults update-data-snapshots update-all-data

$(PY):
	python3 -m venv $(VENV)

help:
	@echo "Targets:"
	@echo "  make install  - install/update python dependencies into venv"
	@echo "  make run      - run Streamlit app"
	@echo "  make format   - run black formatter"
	@echo "  make format-ui - run black on UI/oscillator work files"
	@echo "  make check    - run black --check"
	@echo "  make check-ui - run black --check on UI/oscillator work files"
	@echo "  make compile  - compile Python files"
	@echo "  make test     - run pytest test suite"
	@echo "  make test-ui  - run targeted UI/oscillator tests"
	@echo "  make test-charts - run chart helper tests"
	@echo "  make test-oscillator - run oscillator tests"
	@echo "  make test-sidebar - run sidebar helper tests"
	@echo "  make verify   - run format check, compile, and targeted UI tests"
	@echo "  make update-defaults - recompute and rewrite PowerLaw and LogPeriodic defaults"
	@echo "  make update-data-snapshots - refresh checked-in daily snapshot CSV files"
	@echo "  make update-all-data - refresh snapshots, then recompute checked-in defaults"
	@echo "  make clean    - remove Python cache folders"

install: $(PY)
	$(PIP) install -U pip
	$(PIP) install -r config/requirements.txt

run: $(PY)
	$(STREAMLIT) run app.py

format: $(PY)
	$(BLACK) $(BLACK_CONFIG) $(BLACK_TARGETS)

format-ui: $(PY)
	$(BLACK) $(BLACK_CONFIG) $(UI_CHECK_TARGETS)

check: $(PY)
	$(BLACK) $(BLACK_CONFIG) --check $(BLACK_TARGETS)

check-ui: $(PY)
	$(BLACK) $(BLACK_CONFIG) --check $(UI_CHECK_TARGETS)

compile: $(PY)
	$(PY) -m compileall $(PY_TARGETS)

test: $(PY)
	$(PY) -m pytest -q

test-ui: $(PY)
	$(PY) -m pytest $(UI_TEST_TARGETS) -q

test-charts: $(PY)
	$(PY) -m pytest tests/test_ui_charts_helpers.py -q

test-oscillator: $(PY)
	$(PY) -m pytest tests/test_oscillator.py -q

test-sidebar: $(PY)
	$(PY) -m pytest tests/test_ui_sidebar_helpers.py -q

verify: check-ui compile test-ui

update-defaults: $(PY)
	$(PY) scripts/update_powerlaw_defaults.py

update-data-snapshots: $(PY)
	$(PY) scripts/update_data_snapshots.py

update-all-data: update-data-snapshots update-defaults

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
