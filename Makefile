PYTHON ?= /usr/local/bin/python3

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_DIR := $(ROOT_DIR)/venv
OUTPUT_DIR := $(ROOT_DIR)/output
CONTRIB_DIR := $(ROOT_DIR)/contrib

VENV_ACTIVATE := $(VENV_DIR)/bin/activate

OCV_VER := 3.4.4
OCV_URL := https://github.com/opencv/opencv/archive/$(OCV_VER).tar.gz
OCV_CONTRIB_URL := https://github.com/opencv/opencv_contrib/archive/$(OCV_VER).tar.gz
OCV_ARCHIVE := output/opencv.tar.gz
OCV_CONTRIB_ARCHIVE := output/opencv_contrib.tar.gz
OCV_DIR := $(CONTRIB_DIR)/opencv
OCV_CONTRIB_DIR := $(CONTRIB_DIR)/opencv_contrib

$(VENV_DIR): 
	mkdir -pv $@

$(VENV_ACTIVATE): $(VENV_DIR)
	$(PYTHON) -m venv $(VENV_DIR)
	$(PYTHON) -m pip install -U pip
	touch $@

$(OCV_ARCHIVE):
	test -f $@ || wget -O $@ $(OCV_URL)

$(OCV_CONTRIB_ARCHIVE):
	test -f $@ || wget -O $@ $(OCV_URL)

$(OCV_DIR): $(OCV_ARCHIVE)
	test -d $@ || tar xvzf $< -C $@

$(OCV_CONTRIB_DIR): $(OCV_CONTRIB_ARCHIVE)
	test -d $@ || tar xvzf $< -C $@

opencv: $(OCV_DIR) $(OCV_CONTRIB_DIR)

opencv-contrib:
	wget -O $(OCV_CONTRIB_ARCHIVE) $(OCV_CONTRIB_URL)

setup: $(VENV_ACTIVATE) opencv

run/%:
	source $<
	$(PYTHON) $(notdir $*)
	deactivate