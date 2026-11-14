# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_executable(omniglove_plugin_tests test_hand_conversion.cpp)
target_link_libraries(omniglove_plugin_tests
    PRIVATE Catch2::Catch2WithMain isaac::omniglove_plugin OmnigloveSDK::api
)
catch_discover_tests(omniglove_plugin_tests TEST_PREFIX "omniglove_" ADD_TAGS_AS_LABELS)
