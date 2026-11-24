# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# H.264 NVENC encoder over the device's primary CUDA context.
add_library(nvenc STATIC
    cuda_context.cpp
    nvenc_encoder.cpp
)

set_target_properties(nvenc PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nvenc
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(nvenc
    PUBLIC
        camera_livekit_bridge::video_types
        CUDA::cudart
        CUDA::cuda_driver
    PRIVATE
        nv_codec_headers
        ${CMAKE_DL_LIBS}
)

target_compile_options(nvenc PRIVATE -Wall -Wextra)

add_library(camera_livekit_bridge::nvenc ALIAS nvenc)
