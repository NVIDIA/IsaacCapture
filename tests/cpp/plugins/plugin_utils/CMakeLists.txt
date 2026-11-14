# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(plugin_utils_tests
    test_plugin_device_status_publisher.cpp
    test_wrist_source_config.cpp
)

target_link_libraries(plugin_utils_tests
    PRIVATE
        Catch2::Catch2WithMain
        Teleop::plugin_utils
)

catch_discover_tests(plugin_utils_tests ADD_TAGS_AS_LABELS)
