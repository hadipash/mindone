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

# Set Ascend toolkit path
set(ASCEND_TOOLKIT_DIR /usr/local/Ascend/ascend-toolkit/latest)
set(ASCEND_RUNTIME_DIR /usr/local/Ascend/runtime/latest)

# Allow override from environment
if(DEFINED ENV{ASCEND_TOOLKIT_PATH})
    set(ASCEND_TOOLKIT_DIR $ENV{ASCEND_TOOLKIT_PATH})
endif()

if(DEFINED ENV{ASCEND_RUNTIME_PATH})
    set(ASCEND_RUNTIME_DIR $ENV{ASCEND_RUNTIME_PATH})
endif()

# Find Python and MindSpore using environment variables
if(DEFINED ENV{MINDSPORE_PATH})
    set(MINDSPORE_DIR $ENV{MINDSPORE_PATH})
    message(STATUS "Using MindSpore from: ${MINDSPORE_DIR}")
else()
    # Try to find MindSpore using Python
    find_package(PkgConfig REQUIRED)
    execute_process(
        COMMAND python3 -c "import mindspore; print(mindspore.__file__)"
        OUTPUT_VARIABLE MINDSPORE_PYTHON_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(MINDSPORE_PYTHON_PATH)
        get_filename_component(MINDSPORE_DIR "${MINDSPORE_PYTHON_PATH}" DIRECTORY)
        message(STATUS "Found MindSpore at: ${MINDSPORE_DIR}")
    else()
        message(WARNING "MindSpore not found automatically. Please set MINDSPORE_PATH environment variable.")
        set(MINDSPORE_DIR "")
    endif()
endif()

# Include directories
include_directories(
    ${ASCEND_TOOLKIT_DIR}/include
    ${ASCEND_RUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# If MindSpore directory is provided, add it to include path
if(MINDSPORE_DIR)
    include_directories(${MINDSPORE_DIR})
    include_directories(${MINDSPORE_DIR}/include)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -Wall -std=c++17")

# Source files
set(SOURCES
    block_sparse_attention_ascendc.cc
)

# Create shared library
add_library(block_sparse_attention SHARED ${SOURCES})

# Link Ascend libraries
target_link_libraries(block_sparse_attention
    ${ASCEND_TOOLKIT_DIR}/lib64/libascendcl.so
    ${ASCEND_TOOLKIT_DIR}/lib64/libacl_op_compiler.so
)

# Install
install(TARGETS block_sparse_attention
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

message(STATUS "Ascend Toolkit: ${ASCEND_TOOLKIT_DIR}")
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build configuration complete")
