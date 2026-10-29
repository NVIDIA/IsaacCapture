# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

get_filename_component(CAMERA_STREAMER_ROOT "${ARGUS_SENDER_ROOT}" DIRECTORY)
set(CAMERA_STREAMER_TEST_ROOT "${CAMERA_STREAMER_ROOT}/tests")

set(ARGUS_SENDER_TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CAMERA_STREAMER_TEST_ROOT}/cpp
)

function(argus_sender_add_cpp_test target test_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
    target_include_directories(${target} PRIVATE
        ${ARGUS_SENDER_TEST_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        Threads::Threads
    )
    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "unit"
        TIMEOUT 10
    )
endfunction()

add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/config ${CMAKE_CURRENT_BINARY_DIR}/tests/config)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/geometry_scheduler ${CMAKE_CURRENT_BINARY_DIR}/tests/geometry_scheduler)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/priority_frame_handoff ${CMAKE_CURRENT_BINARY_DIR}/tests/priority_frame_handoff)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/frame_alignment_policy ${CMAKE_CURRENT_BINARY_DIR}/tests/frame_alignment_policy)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/capture_domain_plan ${CMAKE_CURRENT_BINARY_DIR}/tests/capture_domain_plan)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/capture_priority_policy ${CMAKE_CURRENT_BINARY_DIR}/tests/capture_priority_policy)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/avs1_wire ${CMAKE_CURRENT_BINARY_DIR}/tests/avs1_wire)
