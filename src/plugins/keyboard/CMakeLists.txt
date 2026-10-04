# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping keyboard plugin (Linux only)")
    add_custom_target(keyboard_plugin
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping keyboard: Linux only")
    return()
endif()

add_executable(keyboard_plugin
    main.cpp
    keyboard_plugin.cpp
)

target_link_libraries(keyboard_plugin PRIVATE
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
)

install(TARGETS keyboard_plugin RUNTIME DESTINATION plugins/keyboard)
install(FILES plugin.yaml README.md DESTINATION plugins/keyboard)
