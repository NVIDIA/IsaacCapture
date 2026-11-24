# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(camera_livekit_bridge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)

# ==============================================================================
# NVENC headers (ffnvcodec): headers only, loaded dynamically at run time
# ==============================================================================
# Point FETCHCONTENT_SOURCE_DIR_NV_CODEC_HEADERS at an unpacked copy to build offline.
FetchContent_Declare(
    nv_codec_headers
    URL https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/n12.2.72.0.tar.gz
    URL_HASH SHA256=dbeaec433d93b850714760282f1d0992b1254fc3b5a6cb7d76fc1340a1e47563
    SOURCE_SUBDIR no-cmake-here
)
FetchContent_MakeAvailable(nv_codec_headers)
add_library(nv_codec_headers INTERFACE)
target_include_directories(nv_codec_headers SYSTEM INTERFACE "${nv_codec_headers_SOURCE_DIR}/include")

# ==============================================================================
# LiveKit C++ SDK: a prebuilt package (headers and shared libraries)
# ==============================================================================
# Point LIVEKIT_SDK_DIR at an unpacked copy to use another, or to build offline.
if(NOT LIVEKIT_SDK_DIR)
    set(LIVEKIT_SDK_UBUNTU "24.04" CACHE STRING "Ubuntu release the prebuilt SDK was built for")
    if(LIVEKIT_SDK_UBUNTU STREQUAL "22.04")
        set(_livekit_hash "9e568ee6b2bb3c7b0d07faba694930767096a14d0e09422ba01f34077ffb03cb")
    else()
        set(_livekit_hash "5e0feb2e519c6ff131dd2d1c45149115057ae4e21a324fdb249e1789d5614e7b")
    endif()
    FetchContent_Declare(
        livekit_sdk
        URL https://github.com/livekit/client-sdk-cpp/releases/download/v1.12.4/livekit-sdk-ubuntu-${LIVEKIT_SDK_UBUNTU}-x64-1.12.4.tar.gz
        URL_HASH SHA256=${_livekit_hash}
    )
    FetchContent_MakeAvailable(livekit_sdk)
    set(LIVEKIT_SDK_DIR "${livekit_sdk_SOURCE_DIR}")
endif()
find_package(LiveKit CONFIG REQUIRED PATHS "${LIVEKIT_SDK_DIR}" NO_DEFAULT_PATH)
message(STATUS "LiveKit SDK=${LIVEKIT_SDK_DIR}")

# The time-sync library comes from the product tree; a build inside it already has the target.
if(NOT TARGET timesync::timesync)
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH _examples_dir)
    cmake_path(GET _examples_dir PARENT_PATH _repo_dir)
    add_subdirectory(${_repo_dir}/src/core/timesync ${CMAKE_BINARY_DIR}/timesync)
endif()

add_subdirectory(video_types)
add_subdirectory(ipc_client)
add_subdirectory(gpu_ops)
add_subdirectory(nvenc)
add_subdirectory(chrony_clock)
add_subdirectory(livekit_publisher)
add_subdirectory(probe)
add_subdirectory(bridge)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
