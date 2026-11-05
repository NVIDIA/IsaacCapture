# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(revohuman_core_tests
    test_shapes.hpp
    test_device_clock.cpp
    test_device_status.cpp
    test_event_log.cpp
    test_pipeline.cpp
    test_pose_math.cpp
    test_serialize.cpp
    test_sha256.cpp
    test_tactile.cpp
    test_wrist_tracking.cpp
)

target_link_libraries(revohuman_core_tests
    PRIVATE
        Catch2::Catch2WithMain
        revohuman_plugin::revohuman_core
)

catch_discover_tests(revohuman_core_tests)
