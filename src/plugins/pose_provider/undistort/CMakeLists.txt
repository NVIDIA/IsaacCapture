# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Undistortion of one camera image to a distortion-free pinhole; standard library only.
add_library(pose_undistort STATIC
    undistort.cpp
    inc/pose_undistort/undistort.hpp
)

target_include_directories(pose_undistort
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(pose_undistort
    PUBLIC
        pose_provider::pose_engine_api
)

add_library(pose_provider::pose_undistort ALIAS pose_undistort)
