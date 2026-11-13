# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The Pose Provider runtime. It links no engine adapter; each executable links one.
add_library(pose_provider_core STATIC
    pose_provider.cpp
    inc/pose_provider/pose_provider.hpp
)

target_include_directories(pose_provider_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(pose_provider_core
    PUBLIC
        pose_provider::pose_camera_input
        pose_provider::pose_engine_api
    PRIVATE
        log_bridge::log_bridge_core
        pose_provider::pose_undistort
)

add_library(pose_provider::pose_provider_core ALIAS pose_provider_core)
