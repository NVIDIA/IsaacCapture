# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(manus_calibration_tests
    fake_sdk.hpp
    test_engine.cpp
    test_export_writer.cpp
    test_protocol.cpp
    test_sha256.cpp
)

target_link_libraries(manus_calibration_tests
    PRIVATE
        Catch2::Catch2WithMain
        manus_calibration::core
)

# A stable prefix and Catch2 tags as labels, so `ctest -R manus_calibration_` and `ctest -L unit` select the suite.
catch_discover_tests(manus_calibration_tests
    TEST_PREFIX "manus_calibration_"
    ADD_TAGS_AS_LABELS
)
