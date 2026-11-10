# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Video stage: the Orin-to-PC video transport, the PC relay to LiveKit, and an optional PC GPU stage
# ==============================================================================
#   wire/    video_stage_wire   -- AVS1 records, clock exchange, H.264 access-unit helpers, frame correlator
#   sender/  video_stage_sender -- the Orin-side sender of AVS1 records over a direct link
#   app/     video_stage_worker -- the PC relay: receives the Orin's records and publishes them to LiveKit or a file
#
#   core/    video_stage_core   -- (GPU) the operation interface and the example sine-wave warp
#   codec/   video_stage_codec  -- (GPU) NVDEC decoder and NVENC encoder
#
# No CUDA unless BUILD_VIDEO_STAGE_GPU is ON: the relay forwards the Orin's access units untouched. The LiveKit SDK
# (src/livekit_publisher) is needed only by the relay.
#
# Configure with:
#   cmake -B build -DBUILD_VIDEO_STAGE=ON [-DBUILD_VIDEO_STAGE_GPU=ON]
# ==============================================================================

add_subdirectory(wire)
add_subdirectory(sender)

if(BUILD_VIDEO_STAGE_GPU)
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
    add_subdirectory(codec)
endif()

add_subdirectory(app)

if(BUILD_TESTING)
    add_subdirectory(wire/tests)
    add_subdirectory(tests)
endif()
