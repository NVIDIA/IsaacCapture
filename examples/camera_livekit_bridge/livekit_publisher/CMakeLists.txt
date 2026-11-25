# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The LiveKit SDK behind a small interface: join a room, publish a video track, submit H.264 access units, and answer
# the PC's clock-sync requests. No LiveKit type appears in an installed header, and nothing here touches CUDA.
add_library(livekit_publisher STATIC livekit_session.cpp)

target_include_directories(livekit_publisher
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(livekit_publisher
    PUBLIC
        camera_livekit_bridge::video_types
    PRIVATE
        LiveKit::livekit
)

target_compile_options(livekit_publisher PRIVATE -Wall -Wextra)

add_library(camera_livekit_bridge::livekit_publisher ALIAS livekit_publisher)
