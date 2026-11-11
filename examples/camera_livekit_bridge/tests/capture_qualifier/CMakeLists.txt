# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_capture_qualifier test_capture_qualifier.cpp)

target_link_libraries(test_capture_qualifier
    PRIVATE
        camera_livekit_bridge::clock_exchange
)

target_compile_options(test_capture_qualifier PRIVATE -Wall -Wextra)

add_test(NAME capture_qualifier COMMAND test_capture_qualifier)
