# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_sender_clock test_sender_clock.cpp)

target_link_libraries(test_sender_clock
    PRIVATE
        camera_livekit_bridge::sender_clock
        camera_livekit_bridge::video_types
        Threads::Threads
)

target_compile_options(test_sender_clock PRIVATE -Wall -Wextra)

add_test(NAME sender_clock COMMAND test_sender_clock)
