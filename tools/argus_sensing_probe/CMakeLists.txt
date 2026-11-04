# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standalone Argus smoke test. Build and run directly on the Orin:
#   cmake -B build && cmake --build build && ./build/argus_sensing_probe
cmake_minimum_required(VERSION 3.16)
project(argus_sensing_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "Jetson Multimedia API root")
set(NVIDIA_LIBRARY_DIRS
    /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia
    /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra
    /usr/lib/aarch64-linux-gnu/nvidia
    /usr/lib/aarch64-linux-gnu/tegra
)

find_path(ARGUS_INCLUDE_DIR Argus/Argus.h HINTS "${MMAPI_ROOT}/argus/include")
find_library(ARGUS_LIBRARY NAMES nvargus HINTS ${NVIDIA_LIBRARY_DIRS})
find_library(ARGUS_SOCKETCLIENT_LIBRARY NAMES nvargus_socketclient HINTS ${NVIDIA_LIBRARY_DIRS})
find_library(EGL_LIBRARY NAMES EGL)

if(NOT ARGUS_INCLUDE_DIR)
  message(FATAL_ERROR
    "Argus/Argus.h not found under ${MMAPI_ROOT}/argus/include -- "
    "this must build on the Orin with the Jetson Multimedia API installed, not this PC.")
endif()
if(NOT ARGUS_LIBRARY)
  message(FATAL_ERROR "libnvargus not found -- is this actually a Jetson with Argus installed?")
endif()
if(NOT EGL_LIBRARY)
  message(FATAL_ERROR "libEGL not found")
endif()

add_executable(argus_sensing_probe main.cpp)
target_include_directories(argus_sensing_probe PRIVATE ${ARGUS_INCLUDE_DIR})
target_link_libraries(argus_sensing_probe PRIVATE ${ARGUS_LIBRARY} ${EGL_LIBRARY})
if(ARGUS_SOCKETCLIENT_LIBRARY)
  target_link_libraries(argus_sensing_probe PRIVATE ${ARGUS_SOCKETCLIENT_LIBRARY})
endif()
