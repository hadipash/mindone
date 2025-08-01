# Copyright 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14)
project(block_sparse_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MindSpore
find_package(MindSpore REQUIRED)

# Set Ascend toolkit path
set(ASCEND_TOOLKIT_DIR /usr/local/Ascend/ascend-toolkit/latest)
set(ASCEND_RUNTIME_DIR /usr/local/Ascend/runtime/latest)

# Include directories
include_directories(
    ${ASCEND_TOOLKIT_DIR}/include
    ${ASCEND_RUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -Wall")

# Source files
set(SOURCES
    block_sparse_attention_ascendc.cc
)

# Create shared library
add_library(block_sparse_attention SHARED ${SOURCES})

# Link libraries
target_link_libraries(block_sparse_attention
    ${ASCEND_TOOLKIT_DIR}/lib64/libascendcl.so
    ${ASCEND_TOOLKIT_DIR}/lib64/libacl_op_compiler.so
    MindSpore::mindspore
)

# Install
install(TARGETS block_sparse_attention
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
