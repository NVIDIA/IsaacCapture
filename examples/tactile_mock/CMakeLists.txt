# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

# Mock tactile glove for the internal tactile schema; tactile_reader.py reads what it pushes.
add_executable(tactile_pusher
    tactile_pusher.cpp
)

target_link_libraries(tactile_pusher PRIVATE
    pusherio::pusherio
    oxr::oxr_core
    log_bridge::log_bridge_core
    isaacteleop_schema
    ${CMAKE_DL_LIBS}
    Threads::Threads
)

install(TARGETS tactile_pusher
    RUNTIME DESTINATION examples/tactile_mock
)
install(FILES tactile_reader.py
    DESTINATION examples/tactile_mock
)
