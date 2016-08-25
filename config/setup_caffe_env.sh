echo 'Setup Caffe'
export CAFFE_DIR=/home/taritree/software/caffe_larbys
export CAFFE_LIBDIR=${CAFFE_DIR}/build/lib
export CAFFE_INCDIR=${CAFFE_DIR}/build/include
export CAFFE_BINDIR=${CAFFE_DIR}/build/tools
export PATH=${CAFFE_BINDIR}:${PATH}
export LD_LIBRARY_PATH=${CAFFE_LIBDIR}:${LD_LIBRARY_PATH}
export PYTHONPATH=${CAFFE_DIR}/python:${CAFFE_DIR}/python/caffe/proto:${PYTHONPATH}
