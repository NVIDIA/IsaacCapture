# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The camera producer's management channel: an authenticated WebSocket server, the request protocol, the status the
# producer reports, and plain-language camera fault messages. No CUDA or SIPL, so it builds and is tested on a PC.
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
isaac_teleop_third_party_scope_begin()
FetchContent_MakeAvailable(nlohmann_json)
isaac_teleop_third_party_scope_end()

add_library(sensing_control STATIC
    websocket.cpp
    producer_status.cpp
    camera_fault.cpp
    control_handler.cpp
    control_server.cpp
    producer_supervisor.cpp
    inc/sensing_control/websocket.hpp
    inc/sensing_control/producer_status.hpp
    inc/sensing_control/camera_fault.hpp
    inc/sensing_control/control_handler.hpp
    inc/sensing_control/control_server.hpp
    inc/sensing_control/producer_supervisor.hpp
)
add_library(sensing::control ALIAS sensing_control)

set_target_properties(sensing_control PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(sensing_control
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(sensing_control
    PUBLIC
        log_bridge::log_bridge_core
    PRIVATE
        nlohmann_json::nlohmann_json
)

add_subdirectory(mock_producer)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
