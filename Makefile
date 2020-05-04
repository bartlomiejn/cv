PYTHON ?= python3
VENV_NAME ?= cv-python3

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_DIR := venv/$(VENV_NAME)
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

$(VENV_DIR): 
	mkdir -pv $@

$(VENV_ACTIVATE): $(VENV_DIR)
	virtualenv -p python3 $(VENV_NAME)
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .
	touch $@

generate-venv: $(VENV_ACTIVATE)
