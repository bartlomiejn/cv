PYTHON ?= /usr/local/bin/python3
CMAKE ?= cmake

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_DIR := $(ROOT_DIR)/venv
OUTPUT_DIR := $(ROOT_DIR)/output
CONTRIB_DIR := $(ROOT_DIR)/contrib

VENV_ACTIVATE := $(VENV_DIR)/bin/activate
VENV_PYTHON := $(VENV_DIR)/bin/python

OCV_VER := 3.4.4
OCV_URL := https://github.com/opencv/opencv/archive/$(OCV_VER).tar.gz
OCV_CONTRIB_URL := https://github.com/opencv/opencv_contrib/archive/$(OCV_VER).tar.gz
OCV_ARCHIVE := output/opencv.tar.gz
OCV_CONTRIB_ARCHIVE := output/opencv_contrib.tar.gz
OCV_DIR := $(CONTRIB_DIR)/opencv-3.4.4
OCV_CONTRIB_DIR := $(CONTRIB_DIR)/opencv-contrib-3.4.4
OCV_CONTRIB_MODULES := $(OCV_CONTRIB_DIR)/opencv-$(OCV_VER)/modules

$(CONTRIB_DIR):
	mkdir -pv $@

$(OUTPUT_DIR):
	mkdir -pv $@

$(OCV_ARCHIVE): $(OUTPUT_DIR)
	test -f $@ || wget -O $@ $(OCV_URL)

$(OCV_CONTRIB_ARCHIVE): $(OUTPUT_DIR)
	test -f $@ || wget -O $@ $(OCV_URL)

$(OCV_DIR): $(CONTRIB_DIR) $(OCV_ARCHIVE) $(OCV_CONTRIB_ARCHIVE)
	test -d $@ || ( \
		mkdir -pv $@; \
		mkdir -pv $(OCV_CONTRIB_DIR); \
		tar xvzf $(OCV_ARCHIVE) -C $(CONTRIB_DIR); \
		tar xvzf $(OCV_CONTRIB_ARCHIVE) -C $(OCV_CONTRIB_DIR); \
	)

$(VENV_ACTIVATE):
	test -d $(VENV_DIR) || ( \
		mkdir -pv $(VENV_DIR); \
		$(PYTHON) -m venv $(VENV_DIR); \
		$(PYTHON) -m pip install -U pip; \
	)
 
opencv: $(OCV_DIR) $(VENV_ACTIVATE)
	mkdir -pv $(OCV_DIR)/build
	( \
		source $(VENV_ACTIVATE); \
		cd $(OCV_DIR)/build && $(CMAKE) \
			-D CMAKE_BUILD_TYPE=RELEASE \
			-D CMAKE_INSTALL_PREFIX=/usr/local \
			-D OPENCV_EXTRA_MODULES_PATH=$(OCV_CONTRIB_MODULES) \
			-D PYTHON3_LIBRARY=$(shell python pythonlib.py) \
			-D PYTHON3_INCLUDE_DIR=$(shell python include.py) \
			-D PYTHON3_EXECUTABLE=$(VENV_PYTHON) \
			-D BUILD_opencv_python2=OFF \
			-D BUILD_opencv_python3=ON \
			-D INSTALL_PYTHON_EXAMPLES=ON \
			-D INSTALL_C_EXAMPLES=OFF \
			-D OPENCV_ENABLE_NONFREE=ON \
			-D BUILD_EXAMPLES=ON ..; \
	)

venv: $(VENV_ACTIVATE)

setup: $(VENV_ACTIVATE) opencv

run/%: $(VENV_ACTIVATE) $(OUTPUT_DIR)
	( \
		source $(VENV_ACTIVATE); \
		$(PYTHON) $(notdir $*); \
	)	

clean-contrib:
	rm -rf $(CONTRIB_DIR)

clean-output:
	rm -rf $(OUTPUT_DIR)
