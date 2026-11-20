# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(robot_state_publisher_tests
    test_robot_state_publisher.cpp
)

# The publisher writes MCAP records itself; the tests read them back with the MCAP library, whose implementation
# is compiled inside deviceio_session.
target_link_libraries(robot_state_publisher_tests PRIVATE
    robot_state::robot_state_publisher
    deviceio::deviceio_session
    mcap::mcap_core
    Catch2::Catch2WithMain
)

message(STATUS "robot_state_publisher_tests target enabled with Catch2")
# The publisher logs through its own sinks: keep their file in the build tree, and off a forwarding socket an
# interactive shell may be exporting. One value per property: catch_discover_tests re-splits a ';' list into
# further property names, which also drops the SKIP_RETURN_CODE it appends, so a SKIP() would fail.
catch_discover_tests(robot_state_publisher_tests
    PROPERTIES
        ENVIRONMENT "ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs"
        ENVIRONMENT_MODIFICATION "ISAACCAPTURE_LOG_SOCKET=unset:"
)
