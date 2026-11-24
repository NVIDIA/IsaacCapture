# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_notice_queue test_notice_queue.cpp)

target_link_libraries(test_notice_queue
    PRIVATE
        camera_livekit_bridge::notice_queue
)

target_compile_options(test_notice_queue PRIVATE -Wall -Wextra)

add_test(NAME notice_queue COMMAND test_notice_queue)
