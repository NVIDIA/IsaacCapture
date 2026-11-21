# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(no_data_deviceio_session_tests
    test_no_data_session.cpp
)

target_link_libraries(no_data_deviceio_session_tests PRIVATE
    # Whole archive: the replay impls behind NoDataSession need the MCAP symbols that are
    # compiled once inside deviceio_session (mcap_impl.cpp), and nothing in this test
    # references them before the linker has passed that archive.
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,deviceio::deviceio_session>"
    deviceio::deviceio_trackers
    Catch2::Catch2WithMain
)

message(STATUS "no_data_deviceio_session_tests target enabled with Catch2")
catch_discover_tests(no_data_deviceio_session_tests)
