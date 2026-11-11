# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Header-only camera input interface over the engine API types, so a camera adapter can link it without the runtime.
add_library(pose_camera_input INTERFACE)

target_include_directories(pose_camera_input
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(pose_camera_input
    INTERFACE
        pose_provider::pose_engine_api
)

add_library(pose_provider::pose_camera_input ALIAS pose_camera_input)
