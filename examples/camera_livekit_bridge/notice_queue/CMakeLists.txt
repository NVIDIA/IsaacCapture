# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(notice_queue INTERFACE)

target_include_directories(notice_queue
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(notice_queue
    INTERFACE
        camera_livekit_bridge::ipc_client
)

add_library(camera_livekit_bridge::notice_queue ALIAS notice_queue)
