# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(gamepad_plugin_tests
    test_gamepad_connection.cpp
)

target_link_libraries(gamepad_plugin_tests
    PRIVATE
        Catch2::Catch2WithMain
        gamepad::gamepad_core
        SDL3::SDL3-static
)

catch_discover_tests(gamepad_plugin_tests)
