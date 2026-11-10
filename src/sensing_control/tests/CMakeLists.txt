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

# The mock producer end to end: needs the built binaries, so it is skipped (not failed) when they are absent, and it
# needs the PC relay, which exists only in a build with BUILD_VIDEO_STAGE.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND TARGET video_stage_worker)
    add_test(
        NAME sensing_mock_producer_e2e
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_mock_producer_e2e.py
    )
    set_tests_properties(sensing_mock_producer_e2e PROPERTIES
        ENVIRONMENT "SENSING_MOCK_PRODUCER=$<TARGET_FILE:sensing_mock_producer>;VIDEO_STAGE_WORKER=$<TARGET_FILE:video_stage_worker>"
        TIMEOUT 120
    )
endif()
