# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(video_stage_tests
    test_sine_warp.cpp
)

target_link_libraries(video_stage_tests PRIVATE video_stage::core Catch2::Catch2WithMain)

catch_discover_tests(video_stage_tests)

# The receiver is part of the worker executable, not a library, so its source is compiled in directly. It needs
# only POSIX sockets and the AVS1 codec: no CUDA device.
add_executable(video_stage_tcp_receiver_tests
    test_tcp_receiver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../app/tcp_receiver.cpp
)

target_include_directories(video_stage_tcp_receiver_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../app)
target_link_libraries(video_stage_tcp_receiver_tests PRIVATE video_stage::wire Catch2::Catch2WithMain)

catch_discover_tests(video_stage_tcp_receiver_tests)
