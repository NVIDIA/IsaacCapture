# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The Orin-side sender of AVS1 records to the PC video stage: plain POSIX sockets and threads, no CUDA, so the camera
# producer, the mock producer and the tests all link the same code.
add_library(video_stage_sender STATIC
    direct_sender.cpp
    inc/video_stage/direct_sender.hpp
)
add_library(video_stage::sender ALIAS video_stage_sender)

set_target_properties(video_stage_sender PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(video_stage_sender
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(video_stage_sender
    PUBLIC
        video_stage::wire
        log_bridge::log_bridge_core
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
