# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

set(REMOTE_DEVICEIO_SESSION_TEST_SOURCES
    test_remote_session.cpp
)
# Socket tests; remote_link has no socket implementation on Windows.
if(NOT WIN32)
    list(APPEND REMOTE_DEVICEIO_SESSION_TEST_SOURCES test_remote_session_link.cpp)
endif()

add_executable(remote_deviceio_session_tests ${REMOTE_DEVICEIO_SESSION_TEST_SOURCES})

target_link_libraries(remote_deviceio_session_tests PRIVATE
    deviceio::deviceio_session
    deviceio::deviceio_trackers
    mcap::mcap_core
    Catch2::Catch2WithMain
)

message(STATUS "remote_deviceio_session_tests target enabled with Catch2")
# The link logs through its own sinks: keep their file in the build tree, and off a forwarding socket an interactive
# shell may be exporting. One value per property: catch_discover_tests re-splits a ';' list into further
# property names, which also drops the SKIP_RETURN_CODE it appends, so a SKIP() would fail.
catch_discover_tests(remote_deviceio_session_tests
    PROPERTIES
        ENVIRONMENT "ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs"
        ENVIRONMENT_MODIFICATION "ISAACCAPTURE_LOG_SOCKET=unset:"
)
