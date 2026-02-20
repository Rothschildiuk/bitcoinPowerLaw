VENV := venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
BLACK := $(VENV)/bin/black
BLACK_CONFIG := --config config/pyproject.toml

.PHONY: help install run format check test clean

help:
	@echo "Targets:"
	@echo "  make install  - install/update python dependencies into venv"
	@echo "  make run      - run Streamlit app"
	@echo "  make format   - run black formatter"
	@echo "  make check    - run black --check"
	@echo "  make test     - run pytest test suite"
	@echo "  make clean    - remove Python cache folders"

install:
	$(PIP) install -U pip
	$(PIP) install -r config/requirements.txt

run:
	$(STREAMLIT) run app.py

format:
	$(BLACK) $(BLACK_CONFIG) .

check:
	$(BLACK) $(BLACK_CONFIG) --check .

test:
	$(PY) -m pytest -q

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
