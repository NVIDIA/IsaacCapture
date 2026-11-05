# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Video stage: GPU processing of camera video on the PC (NVDEC -> GPU operation -> NVENC)
# ==============================================================================
#   core/   video_stage_core  -- the GPU operation interface and the example sine-wave warp
#   tests/  video_stage_tests -- Catch2 tests; need a CUDA-capable GPU
#
# Configure with:
#   cmake -B build -DBUILD_VIDEO_STAGE=ON
# ==============================================================================

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Real code for the common desktop architectures plus PTX for 90, which newer GPUs JIT-compile.
    set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89-real;90")
endif()

find_package(CUDAToolkit REQUIRED)

# NVDEC/NVENC API headers (MIT licence) with run-time loaders for libnvcuvid and libnvidia-encode, pinned by hash.
include(FetchContent)
FetchContent_Declare(
    nv_codec_headers
    URL https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/n12.2.72.0.tar.gz
    URL_HASH SHA256=dbeaec433d93b850714760282f1d0992b1254fc3b5a6cb7d76fc1340a1e47563
)
FetchContent_GetProperties(nv_codec_headers)
if(NOT nv_codec_headers_POPULATED)
    FetchContent_Populate(nv_codec_headers)
endif()
add_library(video_stage_nv_codec_headers INTERFACE)
target_include_directories(video_stage_nv_codec_headers SYSTEM INTERFACE "${nv_codec_headers_SOURCE_DIR}/include")

add_subdirectory(core)
add_subdirectory(wire)
add_subdirectory(codec)
add_subdirectory(app)

if(BUILD_TESTING)
    add_subdirectory(tests)
    add_subdirectory(wire/tests)
endif()
