# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(teleop_ros2_mcap_generator
    mcap_generator.cpp
)

target_link_libraries(teleop_ros2_mcap_generator
    PRIVATE
        deviceio::deviceio_session
        mcap::mcap_core
)

install(TARGETS teleop_ros2_mcap_generator
    RUNTIME DESTINATION integrations/teleop_ros2/cpp/integration_tests
)
