#* Variables
SHELL := /usr/bin/env bash

THIS_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
REPO_DIR := $(patsubst %/,%,$(dir $(THIS_MAKEFILE)))
REPO_PREFIX := ../$(notdir $(abspath $(REPO_DIR)))

PYTHON := python3
PYTHONPATH := $(REPO_DIR)/src

CXX_SOURCES = $(shell find $(REPO_DIR) -name '*.cpp' -o -name '*.cxx' -o -name '*.cc' -o -name '*.c++' -o -name '*.hpp' -o -name '*.h')

#* Makefile debugging
print-%: ; @$(warning $* is $($*) ($(value $*)) (from $(origin $*)))

define message
@echo -n "make[top]: "
@echo $(1)
endef

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

#* Installation
.PHONY: install
install:
	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n
	-poetry run mypy --install-types --non-interactive ./

.PHONY: install-conda-deps install-conda-deps-manually
install-conda-deps:
	conda env create --file conda.yaml

install-conda-deps-manually:
	conda create -n py38 python=3.8.10
	conda activate py38
	python3 -m pip install --upgrade pip
	python3 -m pip install pybind11 
	python3 -m pip install tensorflow==2.10.1
	conda env export | tee conda.yaml

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle format-cpp

.PHONY: format-cpp
format-cpp:
	clang-format -i -style=file $(CXX_SOURCES)

#* Linting
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov-report=html --cov=PyCXpress tests/
	poetry run coverage-badge -o assets/images/coverage.svg -f

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./
	poetry run darglint --verbosity 2 PyCXpress tests

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report
	poetry run bandit -ll --recursive PyCXpress tests

.PHONY: lint
lint: test check-codestyle mypy check-safety

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D bandit@latest darglint@latest "isort[colors]@latest" mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	poetry add -D --allow-prereleases black@latest

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-dist
build-dist:
	$(PYTHON) -m build --outdir dist/

.PHONY: build-remove
build-remove:
	rm -rf dist/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove
