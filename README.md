# cv

Usage of classical computer vision techniques or machine/deep learning for:
- Image processing operations
- Feature extraction
- Object recognition 
- Classification

### Building (macOS)

Required packages: `cmake pkg-config wget jpeg libpng libtiff openexr eigen tbb hdf5`

Venv & OpenCV setup: `make setup`

### Building (linux)

`sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"`

`build-essential checkinstall cmake pkg-config yasm gfortran libjpeg8-dev libpng-dev libjasper1 libjasper-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk2.0-dev libtbb-dev qt5-default libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrnb-dev libavresample-dev x264 v4l-utils libeigen3-dev libhdf5-dev`

### Running

`make run SRC={script path} PARAMS={optional params}`

