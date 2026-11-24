# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping DAYANGlove: local IPC and clock mapping are Linux-only")
    return()
endif()

add_subdirectory(format)
add_subdirectory(reader)
add_executable(dayanglove_plugin main.cpp)
target_link_libraries(dayanglove_plugin PRIVATE pusherio::pusherio oxr::oxr_core dayanglove::format)

set(plugin_files source.py receiver.py plugin.yaml README.md requirements.txt)
set(protocol_files
    protocol/glove_frame.proto
    protocol/glove_frame.proto.license
    protocol/glove_frame_pb2.py
    protocol/glove_frame_pb2.py.license
)
foreach(file IN LISTS plugin_files protocol_files)
    configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()
configure_file(
    "${CMAKE_SOURCE_DIR}/LICENSES/Apache-2.0.txt"
    "${CMAKE_CURRENT_BINARY_DIR}/protocol/LICENSE-Apache-2.0.txt"
    COPYONLY
)

install(TARGETS dayanglove_plugin RUNTIME DESTINATION plugins/dayanglove COMPONENT dayanglove)
install(FILES ${plugin_files} DESTINATION plugins/dayanglove COMPONENT dayanglove)
install(FILES ${protocol_files} "${CMAKE_CURRENT_BINARY_DIR}/protocol/LICENSE-Apache-2.0.txt"
    DESTINATION plugins/dayanglove/protocol COMPONENT dayanglove)
