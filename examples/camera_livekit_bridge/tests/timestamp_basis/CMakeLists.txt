# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_timestamp_basis test_timestamp_basis.cpp)

target_link_libraries(test_timestamp_basis
    PRIVATE
        camera_livekit_bridge::video_types
)

target_compile_options(test_timestamp_basis PRIVATE -Wall -Wextra)

add_test(NAME timestamp_basis COMMAND test_timestamp_basis)
