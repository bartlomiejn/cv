PYTHON ?= /usr/local/bin/python3
CMAKE ?= cmake

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
OCV_DIR := $(CONTRIB_DIR)/opencv-3.4.4

$(CONTRIB_DIR):
	mkdir -pv $@

$(OUTPUT_DIR):
	mkdir -pv $@

$(OCV_ARCHIVE):
	test -f $@ || wget -O $@ $(OCV_URL)

$(OCV_CONTRIB_ARCHIVE):
	test -f $@ || wget -O $@ $(OCV_URL)

$(OCV_DIR): $(CONTRIB_DIR) $(OCV_ARCHIVE) $(OCV_CONTRIB_ARCHIVE)
	test -d $@ || ( \
		mkdir -pv $@; \
		tar xvzf $(OCV_ARCHIVE) -C $(CONTRIB_DIR); \
		tar xvzf $(OCV_CONTRIB_ARCHIVE) -C $(CONTRIB_DIR) \
	)

$(VENV_ACTIVATE):
	test -d $(VENV_DIR) || ( \
		mkdir -pv $(VENV_DIR)
		$(PYTHON) -m venv $(VENV_DIR) \
		$(PYTHON) -m pip install -U pip \
		touch $@
	)

opencv: $(OCV_DIR) 
	mkdir -pv $(OCV_DIR)/build
	source $(VENV_ACTIVATE)
	cd $(OCV_DIR)/build && $(CMAKE) \
		-D CMAKE_BUILD_TYPE=RELEASE \
    	-D CMAKE_INSTALL_PREFIX=/usr/local \
    	-D OPENCV_EXTRA_MODULES_PATH=$(OCV_DIR)/modules \
    	-D PYTHON3_LIBRARY=`python -c 'import subprocess; import sys; s = subprocess.check_output("python-config --configdir", shell=True).decode("utf-8").strip(); (M, m) = sys.version_info[:2]; print("{}/libpython{}.{}.dylib".format(s, M, m))'` \
    	-D PYTHON3_INCLUDE_DIR=`python -c 'import distutils.sysconfig as s; print(s.get_python_inc())'` \
    	-D PYTHON3_EXECUTABLE=$(VIRTUAL_ENV)/bin/python \
    	-D BUILD_opencv_python2=OFF \
    	-D BUILD_opencv_python3=ON \
    	-D INSTALL_PYTHON_EXAMPLES=ON \
    	-D INSTALL_C_EXAMPLES=OFF \
    	-D OPENCV_ENABLE_NONFREE=ON \
    	-D BUILD_EXAMPLES=ON ..
	deactivate

setup: $(VENV_ACTIVATE) opencv

run/%: $(VENV_ACTIVATE) (OUTPUT_DIR)
	source $(VENV_ACTIVATE)
	$(PYTHON) $(notdir $*)
	deactivate

clean-contrib:
	rm -rf $(CONTRIB_DIR)

clean-output:
	rm -rf $(OUTPUT_DIR)
