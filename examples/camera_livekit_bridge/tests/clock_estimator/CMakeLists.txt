# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_clock_estimator test_clock_estimator.cpp)

target_link_libraries(test_clock_estimator
    PRIVATE
        camera_livekit_bridge::clock_exchange
)

target_compile_options(test_clock_estimator PRIVATE -Wall -Wextra)

add_test(NAME clock_estimator COMMAND test_clock_estimator)
