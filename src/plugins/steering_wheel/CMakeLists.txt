# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping steering_wheel plugin (Linux only)")
    add_custom_target(steering_wheel_plugin
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping steering_wheel: Linux only")
    return()
endif()

add_executable(steering_wheel_plugin
    main.cpp
    steering_wheel_plugin.cpp
)

target_link_libraries(steering_wheel_plugin PRIVATE
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
)

install(TARGETS steering_wheel_plugin RUNTIME DESTINATION plugins/steering_wheel)
install(FILES plugin.yaml README.md DESTINATION plugins/steering_wheel)
