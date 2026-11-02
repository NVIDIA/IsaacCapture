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

catch_discover_tests(manus_calibration_tests)
