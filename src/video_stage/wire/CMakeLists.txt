# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The AVS1 record codec: plain C++, no CUDA, so the Orin-side sender can link it too.
add_library(video_stage_wire STATIC
    avs1.cpp
    h264_sps.cpp
)
add_library(video_stage::wire ALIAS video_stage_wire)

set_target_properties(video_stage_wire PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(video_stage_wire
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)
