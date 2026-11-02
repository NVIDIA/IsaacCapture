# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Header-only engine interface; standard library only, so any adapter can link it.
add_library(pose_engine_api INTERFACE)

target_include_directories(pose_engine_api
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

add_library(pose_provider::pose_engine_api ALIAS pose_engine_api)
