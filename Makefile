.DEFAULT_GOAL := example-tensorflow

#*****************************************#
#*               VARIABLES               *#
#*****************************************#

THIS_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
THIS_MAKEFILE_DIR := $(patsubst %/,%,$(dir $(THIS_MAKEFILE)))
REPO_DIR := $(THIS_MAKEFILE_DIR)
REPO_PREFIX := ../$(notdir $(abspath $(REPO_DIR)))

THIRD_PARTY_DIR := $(realpath $(REPO_DIR)/third_party)
ifeq ($(CPM_SOURCE_CACHE),)
  CPM_SOURCE_CACHE := $(THIRD_PARTY_DIR)
endif

POETRY := poetry
PYTHON := $(POETRY) run python3
PYTHONPATH := $(REPO_DIR)/src
CMAKE := cmake

ENVS := CC=clang CXX=clang++
ENVS += CPM_SOURCE_CACHE=$(CPM_SOURCE_CACHE)
ENVS += POETRY_VIRTUALENVS_IN_PROJECT=true
ENVS += PATH=$(REPO_DIR)/.venv/bin:$(PATH)
ENVS += LD_LIBRARY_PATH=$(THIRD_PARTY_DIR)/libtensorflow_cc/lib:$(LD_LIBRARY_PATH)

SHELL := /usr/bin/env $(ENVS) bash

#*****************************************#
#*               UTILITIES               *#
#*****************************************#
#* Makefile debugging
print-%: ; @$(warning $* is $($*) ($(value $*)) (from $(origin $*)))
define message
@echo -n "make[top]: "
@echo $(1)
endef


#*****************************************#
#*             QUICK COMMANDS            *#
#*****************************************#
.PHONY: example example-pycxpress example-tensorflow
example-pycxpress:
	env -C $(REPO_DIR)/src/PyCXpress/example $(POETRY) run make run

example-graph: ./sample/saved_model/saved_model.pb
./sample/saved_model/saved_model.pb: ./sample/main.py
	env -C $(THIS_MAKEFILE_DIR)/sample --unset=LD_LIBRARY_PATH $(PYTHON) main.py

example-tensorflow: build-sample example-graph
	$(call message, Run ./build/sample/sample)
	@env TF_CPP_MIN_LOG_LEVEL=2 $(THIS_MAKEFILE_DIR)/build/sample/sample --name whole_flow -- ./sample/saved_model/

example: example-pycxpress example-tensorflow

build: build-sample build-dist
rebuild: cleanup source-all build
.NOTPARALLEL: rebuild

#*****************************************#
#*                OPTIONS                *#
#*****************************************#

CMAKE_OPTIONS :=

USE_LIBTENSORFLOW_CC := 0
ifeq ($(USE_LIBTENSORFLOW_CC),1)
	ifeq ($(CMAKE_OPTIONS),)
		CMAKE_OPTIONS += -DCMAKE_PREFIX_PATH="$(REPO_DIR)/third_party/libtorch;$(REPO_DIR)/third_party/libtensorflow_cc"
	endif
else
	ifeq ($(CMAKE_OPTIONS),)
		CMAKE_OPTIONS += -DCMAKE_PREFIX_PATH="$(REPO_DIR)/third_party/protobuf"
	endif
endif
CMAKE_OPTIONS += -DUSE_LIBTENSORFLOW_CC=$(USE_LIBTENSORFLOW_CC)

ifneq ($(TYPE),)
  BUILD_TYPE_R := Release
  BUILD_TYPE_D := Debug
  BUILD_TYPE_RD := RelWithDebInfo
  BUILD_TYPE_MR := MinSizeRel

  CMAKE_OPTIONS += -DCMAKE_BUILD_TYPE=$(BUILD_TYPE_$(TYPE))
endif

ifneq ($(COVERAGE),)
  CODE_COVERAGE_OPTIONS := 0 1
  CMAKE_OPTIONS += -DENABLE_TEST_COVERAGE=$(COVERAGE)
endif

EMPTY :=
ifneq ($(SANITIZER),)
  SANITIZER_OPTIONS := Address Memory MemoryWithOrigins Undefined Thread Leak
  CMAKE_OPTIONS += -DUSE_SANITIZER=$(subst $(empty) $(empty),;,$(SANITIZER))
endif

ifneq ($(STATIC_CHECK),)
  STATIC_CHECK_OPTIONS := clang-tidy iwyu cppcheck
  CMAKE_OPTIONS += $(foreach TYPE,$(STATIC_CHECK),-DUSE_STATIC_ANALYZER=$(TYPE))
endif

ifneq ($(CCACHE),)
  CCACHE_OPTIONS := ON OFF
  CMAKE_OPTIONS += -DUSE_CCACHE=$(CCACHE)
endif


#*****************************************#
#*                ACTIONS                *#
#*****************************************#

#* CMake
source-all: FORCE
	$(call message, Clean)
	rm -rf build
	$(call message, Source)
	$(CMAKE) -B build $(CMAKE_OPTIONS)

source-sample: FORCE
	$(call message, Source)
	$(CMAKE) -S sample -B build/sample $(CMAKE_OPTIONS)

build-sample: source-sample
	$(call message, Build)
	$(CMAKE) --build build/sample

doc: FORCE
	$(call message, Source)
	$(CMAKE) -S doc -B build/doc $(CMAKE_OPTIONS)
	$(call message, GenerateDocs)
	$(CMAKE) --build build/doc --target GenerateDocs


#* Protobuf
.PHONY: protobuf-install
protobuf-install:
	mkdir -p $(THIRD_PARTY_DIR)
	-env -C $(THIRD_PARTY_DIR) git clone -b v3.9.2 https://github.com/protocolbuffers/protobuf protobuf.src
	env -C $(THIRD_PARTY_DIR)/protobuf.src cmake -S cmake -B build -DCMAKE_INSTALL_PREFIX=$(THIRD_PARTY_DIR)/protobuf -Dprotobuf_BUILD_TESTS=OFF
	env -C $(THIRD_PARTY_DIR)/protobuf.src cmake --build build --target install

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | python3 -
	~/.local/share/pypoetry/venv/bin/pip install poetry-plugin-export

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | python3 - --uninstall

#* Installation
.PHONY: install
install:
	$(POETRY) lock --no-update -n
	$(POETRY) export --without-hashes > requirements.txt
	$(POETRY) export -E tensorflow --without-hashes > requirements.tensorflow.txt
	$(POETRY) install -n --extras tensorflow
	-$(POETRY) run mypy --install-types --non-interactive ./src

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
	$(POETRY) run pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	$(POETRY) run pyupgrade --exit-zero-even-if-changed --py38-plus sample/*.py src/**/*.py tests/**/*.py
	$(POETRY) run isort --settings-path pyproject.toml src sample tests
	$(POETRY) run black --config pyproject.toml --extend-exclude third_party ./

.PHONY: formatting
formatting: codestyle format-cpp format-cmake

.PHONY: format-cpp
EXTRA_CXX_SOURCES += $(shell find $(REPO_DIR)/src/PyCXpress -name '*.cpp' -o -name '*.cxx' -o -name '*.cc' -o -name '*.c++' -o -name '*.hpp' -o -name '*.h')
format-cpp:
	clang-format -i -style=file $(EXTRA_CXX_SOURCES)

.PHONY: format-cmake
#[Format.$(CMAKE)](https://github.com/TheLartians/Format.cmake)
format-cmake:
	$(call message, Source CMAKE)
	$(CMAKE) -B build $(CMAKE_OPTIONS)
	$(call message, Format)
	$(CMAKE) --build build --target fix-format
	$(call message, Revert back all those tensorflow proto changes because of clang-format bug)
	-git checkout src/TensorflowCpy/proto/tensorflow/

#* Linting
.PHONY: test test-pycxpress test-tfcpy
test-pycxpress:
	PYTHONPATH=$(PYTHONPATH) $(POETRY) run pytest -c pyproject.toml --cov-report=html --cov=PyCXpress tests/
	$(POETRY) run coverage-badge -o assets/images/coverage.svg -f

test-tfcpy:
	$(call message, Source)
	$(CMAKE) -S tests -B build/tests $(CMAKE_OPTIONS)
	$(call message, Build)
	$(CMAKE) --build build/tests
	$(call message, Run ./build/tests/TensorflowCpyTests for test)
	CTEST_OUTPUT_ON_FAILURE=1 $(CMAKE) --build build/tests --target test

test: test-pycxpress test-tfcpy

.PHONY: check-codestyle
check-codestyle:
	$(POETRY) run isort --diff --check-only --settings-path pyproject.toml src sample tests
	$(POETRY) run black --diff --check --config pyproject.toml --extend-exclude third_party ./
	$(POETRY) run darglint --verbosity 2 src sample tests

.PHONY: mypy
mypy:
	$(POETRY) run mypy --config-file pyproject.toml ./src

.PHONY: check-safety
check-safety:
	$(POETRY) check
	-$(POETRY) run safety check --full-report
	$(POETRY) run bandit -ll --recursive src/PyCXpress tests

.PHONY: lint
lint: test check-codestyle mypy check-safety

.PHONY: update-dev-deps
update-dev-deps:
	$(POETRY) add -D bandit@latest darglint@latest "isort[colors]@latest" mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	$(POETRY) add -D --allow-prereleases black@latest

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
build-remove: cleanup
	test ! -d build/sample || cmake --build build/sample --target clean
	test ! -d build/tests  || cmake --build build/tests --target clean
	test ! -d build/doc    || cmake --build build/doc --target clean

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove build-remove

.PHONY: distclean
distclean: cleanup
	rm -rf $(REPO_DIR)/dist/
	rm -rf $(REPO_DIR)/build/
	rm -rf $(REPO_DIR)/sample/frozen_graph $(REPO_DIR)/sample/saved_model

FORCE:
