# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Measures the sender's clock over its UDP clock responder.
add_library(sender_clock STATIC sender_clock_client.cpp)

target_include_directories(sender_clock
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(sender_clock
    PUBLIC
        camera_livekit_bridge::clock_exchange
    PRIVATE
        camera_livekit_bridge::video_types
        Threads::Threads
)

target_compile_options(sender_clock PRIVATE -Wall -Wextra)

add_library(camera_livekit_bridge::sender_clock ALIAS sender_clock)
