# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualifies a sender CLOCK_REALTIME capture time against chrony's evidence for both hosts.
add_library(chrony_clock STATIC chrony_capture_clock.cpp)

target_include_directories(chrony_clock
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(chrony_clock
    PUBLIC
        timesync::timesync
)

target_compile_options(chrony_clock PRIVATE -Wall -Wextra)

add_library(camera_livekit_bridge::chrony_clock ALIAS chrony_clock)
