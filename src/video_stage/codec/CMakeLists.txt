# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NVDEC decoder and NVENC encoder adapters over a shared CUDA context.
add_library(video_stage_codec STATIC
    cuda_context.cpp
    nvdec_decoder.cpp
    nvenc_encoder.cpp
)
add_library(video_stage::codec ALIAS video_stage_codec)

set_target_properties(video_stage_codec PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(video_stage_codec
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(video_stage_codec
    PUBLIC
        video_stage::core
        CUDA::cudart
        CUDA::cuda_driver
    PRIVATE
        video_stage_nv_codec_headers
        ${CMAKE_DL_LIBS}
)
