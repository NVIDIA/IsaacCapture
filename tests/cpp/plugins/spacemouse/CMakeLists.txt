# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(spacemouse_plugin_tests
    spacemouse_test_dirs.hpp
    test_device_discovery.cpp
    test_hid_report.cpp
    test_spacemouse_connection.cpp
)

target_link_libraries(spacemouse_plugin_tests
    PRIVATE
        Catch2::Catch2WithMain
        spacemouse::spacemouse_core
)

catch_discover_tests(spacemouse_plugin_tests)
