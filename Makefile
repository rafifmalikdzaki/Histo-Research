# Quality gates for the histodae package.
#
#   make check     lint + typecheck + tests  (what CI should run)
#   make lint      ruff check
#   make typecheck mypy over the typed boundaries
#   make test      pytest
#   make build     build the wheel
#
# `ruff format` is deliberately NOT part of the gate: the tree has never been
# format-clean and reformatting 50+ files would bury the real changes.

PYTHON ?= .venv/bin/python
RUFF   ?= ruff
MYPY   ?= $(PYTHON) -m mypy
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help check lint typecheck test build wheel install-dev clean

help:
	@echo "make check      run lint + typecheck + tests"
	@echo "make lint       ruff check"
	@echo "make typecheck  mypy over the typed boundaries"
	@echo "make test       pytest"
	@echo "make build      build the sdist + wheel"

check: lint typecheck test

lint:
	$(RUFF) check src tests scripts joiv_revision/scripts *.py

typecheck:
	$(MYPY)

test:
	$(PYTEST) tests/

build wheel:
	uv build

install-dev:
	uv pip install -e ".[dev]"

clean:
	rm -rf dist build .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -not -path './.venv/*' -exec rm -rf {} +
