# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Socket-based tests; remote_link has no socket implementation on Windows.
if(WIN32)
    return()
endif()

add_executable(remote_link_tests
    test_clock_offset.cpp
    test_link.cpp
    test_message_header.cpp
    test_reject.cpp
    test_tcp_transport.cpp
    test_websocket_transport.cpp
)

target_link_libraries(remote_link_tests PRIVATE
    remote_link::remote_link
    oxr::oxr_utils
    Catch2::Catch2WithMain
)

message(STATUS "remote_link_tests target enabled with Catch2")
# Link logs through its own sinks: keep their file in the build tree, and off a forwarding socket an interactive
# shell may be exporting. One value per property: catch_discover_tests re-splits a ';' list into further
# property names, which also drops the SKIP_RETURN_CODE it appends, so a SKIP() would fail.
catch_discover_tests(remote_link_tests ADD_TAGS_AS_LABELS
    PROPERTIES
        ENVIRONMENT "ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs"
        ENVIRONMENT_MODIFICATION "ISAACCAPTURE_LOG_SOCKET=unset:"
)
