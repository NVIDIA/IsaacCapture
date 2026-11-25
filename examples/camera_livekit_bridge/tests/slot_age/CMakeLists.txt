# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_slot_age test_slot_age.cpp)

target_link_libraries(test_slot_age
    PRIVATE
        camera_livekit_bridge::video_types
)

target_compile_options(test_slot_age PRIVATE -Wall -Wextra)

add_test(NAME slot_age COMMAND test_slot_age)
