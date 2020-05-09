PYTHON ?= /usr/local/bin/python3
CMAKE ?= cmake
OCV_VER ?= 4.3.0
JLEVEL ?= 10

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_DIR := $(ROOT_DIR)/venv
OUTPUT_DIR := $(ROOT_DIR)/output

VENV_ACTIVATE := $(VENV_DIR)/bin/activate
VENV_PYTHON := $(VENV_DIR)/bin/python

OCV_URL := https://github.com/opencv/opencv/archive/$(OCV_VER).tar.gz
OCV_CONTRIB_URL := https://github.com/opencv/opencv_contrib/archive/$(OCV_VER).tar.gz
OCV_ARCHIVE := output/opencv.tar.gz
OCV_CONTRIB_ARCHIVE := output/opencv_contrib.tar.gz
OCV_DIR := $(OUTPUT_DIR)/opencv-$(OCV_VER)
OCV_CONTRIB_DIR := $(OUTPUT_DIR)/opencv-contrib-$(OCV_VER)
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
		tar -xzf $(OCV_ARCHIVE) -C $(OUTPUT_DIR); \
		tar -xzf $(OCV_CONTRIB_ARCHIVE) -C $(OCV_CONTRIB_DIR); \
	)

$(VENV_ACTIVATE):
	test -d $(VENV_DIR) || ( \
		mkdir -pv $(VENV_DIR); \
		$(PYTHON) -m venv $(VENV_DIR); \
		$(PYTHON) -m pip install -U pip; \
	)
 
opencv: $(OCV_DIR) $(VENV_ACTIVATE)
	mkdir -pv $(OCV_DIR)/build
	mkdir -pv $(OUTPUT_DIR)/obj-opencv-$(OCV_VER)
	cd $(OCV_DIR)/build && $(CMAKE) \
		-DCMAKE_BUILD_TYPE=RELEASE \
		-DCMAKE_INSTALL_PREFIX=$(OUTPUT_DIR)/obj-opencv-$(OCV_VER) \
		-DPYTHON3_LIBRARY=$(shell $(VENV_PYTHON) pythonlib.py) \
		-DPYTHON3_INCLUDE_DIR=$(shell $(VENV_PYTHON) include.py) \
		-DPYTHON3_EXECUTABLE=$(VENV_PYTHON) \
		-DBUILD_opencv_python2=OFF \
		-DBUILD_opencv_python3=ON \
		-DINSTALL_PYTHON_EXAMPLES=ON \
		-DINSTALL_C_EXAMPLES=OFF \
		-DOPENCV_ENABLE_NONFREE=ON \
		-DBUILD_EXAMPLES=ON \
		.. 
	cd $(OCV_DIR)/build && $(MAKE) -j$(JLEVEL)

venv: $(VENV_ACTIVATE)

setup: venv opencv

run/%: $(VENV_ACTIVATE) $(OUTPUT_DIR)
	( \
		source $(VENV_ACTIVATE); \
		$(PYTHON) $(notdir $*); \
	)

clean-venv:
	rm -rf $(VENV_DIR)

clean-output:
	rm -rf $(OUTPUT_DIR)
