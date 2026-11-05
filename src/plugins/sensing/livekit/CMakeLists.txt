# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The LiveKit SDK behind a small interface: join a room, publish a video track, submit NV12 frames.
# No LiveKit type appears in an installed header, and nothing here touches CUDA.
add_library(sensing_livekit STATIC
    livekit_session.cpp
    inc/sensing_livekit/livekit_session.hpp
    inc/sensing_livekit/video_track_sink.hpp
)

target_include_directories(sensing_livekit
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(sensing_livekit
    PUBLIC
        log_bridge::log_bridge_core
    PRIVATE
        LiveKit::livekit
)

add_library(sensing::livekit ALIAS sensing_livekit)
