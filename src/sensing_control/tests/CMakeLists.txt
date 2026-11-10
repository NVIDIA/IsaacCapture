# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(sensing_control_tests
    test_websocket.cpp
    test_camera_fault.cpp
    test_driver_log_capture.cpp
    test_producer_status.cpp
    test_control_handler.cpp
    test_producer_supervisor.cpp
    test_control_server.cpp
)

target_link_libraries(sensing_control_tests
    PRIVATE
        sensing::control
        nlohmann_json::nlohmann_json
        Catch2::Catch2WithMain
)

catch_discover_tests(sensing_control_tests)
