# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(video_stage_core STATIC
    sine_warp.cu
)
add_library(video_stage::core ALIAS video_stage_core)

set_target_properties(video_stage_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(video_stage_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(video_stage_core
    PUBLIC
        CUDA::cudart
)
