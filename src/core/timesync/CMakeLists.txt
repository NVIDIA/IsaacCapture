# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Chrony evidence parsing, wall-to-monotonic conversion and capture qualification. Standard library and POSIX
# clocks only, so it also configures on its own (cmake -S src/core/timesync) and builds without CUDA, LiveKit
# or Python.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(isaaccapture_timesync LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)
endif()

add_subdirectory(cpp)

# Standalone configure runs the module's tests; in the product tree tests/cpp/core adds them.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND BUILD_TESTING)
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH _core_dir)
    cmake_path(GET _core_dir PARENT_PATH _src_dir)
    cmake_path(GET _src_dir PARENT_PATH _repo_dir)
    add_subdirectory(${_repo_dir}/tests/cpp/core/timesync tests)
endif()
