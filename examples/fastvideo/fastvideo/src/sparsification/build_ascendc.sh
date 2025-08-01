#!/bin/bash

# Build script for BlockSparseAttention Ascend C kernel
# This script compiles the Ascend C kernel for Ascend 910B

set -e

# Configuration
ASCEND_TOOLKIT_PATH=${ASCEND_TOOLKIT_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
ASCEND_RUNTIME_PATH=${ASCEND_RUNTIME_PATH:-/usr/local/Ascend/runtime/latest}
MINDSPORE_PATH=${MINDSPORE_PATH:-/usr/local/python3.7/site-packages/mindspore}

# Check if Ascend toolkit is available
if [ ! -d "$ASCEND_TOOLKIT_PATH" ]; then
    echo "Error: Ascend toolkit not found at $ASCEND_TOOLKIT_PATH"
    echo "Please set ASCEND_TOOLKIT_PATH environment variable"
    exit 1
fi

# Create build directory
BUILD_DIR="build_ascendc"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Set environment variables
export ASCEND_HOME=$ASCEND_TOOLKIT_PATH
export ASCEND_RUNTIME_DIR=$ASCEND_RUNTIME_PATH
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_PATH/lib64:$ASCEND_RUNTIME_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$ASCEND_TOOLKIT_PATH/bin:$PATH

# Configure and build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DMINDSPORE_PATH=$MINDSPORE_PATH \
    -DASCEND_TOOLKIT_DIR=$ASCEND_TOOLKIT_PATH \
    -DASCEND_RUNTIME_DIR=$ASCEND_RUNTIME_PATH

# Build
make -j$(nproc)

# Install
make install

echo "Build completed successfully!"
echo "Ascend C kernel built and installed in ./install"
