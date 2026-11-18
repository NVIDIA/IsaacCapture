# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(sharpa_avatar_plugin_tests
    test_avatar_health.cpp
)

target_link_libraries(sharpa_avatar_plugin_tests
    PRIVATE
        Catch2::Catch2WithMain
        isaac::avatar_plugin
)

catch_discover_tests(sharpa_avatar_plugin_tests)
