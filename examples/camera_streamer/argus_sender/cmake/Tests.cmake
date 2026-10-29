# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

get_filename_component(CAMERA_STREAMER_ROOT "${ARGUS_SENDER_ROOT}" DIRECTORY)
set(CAMERA_STREAMER_TEST_ROOT "${CAMERA_STREAMER_ROOT}/tests")

set(ARGUS_SENDER_TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CAMERA_STREAMER_TEST_ROOT}/cpp
    ${ARGUS_INCLUDE_DIR}
)

function(argus_sender_add_cpp_test target test_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
    target_include_directories(${target} PRIVATE
        ${ARGUS_SENDER_TEST_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        ${ARGUS_LINK_LIBRARY}
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

if(ARGUS_SENDER_CPU_TESTS_ONLY)
    return()
endif()

add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/nv_remap ${CMAKE_CURRENT_BINARY_DIR}/tests/nv_remap)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/rate_limiter ${CMAKE_CURRENT_BINARY_DIR}/tests/rate_limiter)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/alignment_policy ${CMAKE_CURRENT_BINARY_DIR}/tests/alignment_policy)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/bundle_assembler ${CMAKE_CURRENT_BINARY_DIR}/tests/bundle_assembler)
add_subdirectory(${CAMERA_STREAMER_TEST_ROOT}/cpp/tcp_transport ${CMAKE_CURRENT_BINARY_DIR}/tests/tcp_transport)

option(ARGUS_SENDER_HARDWARE_TESTS
       "Enable tests that capture from physical stereo cameras"
       OFF)
set(ARGUS_SENDER_HARDWARE_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/hardware_stereo_smoke.yaml"
    CACHE FILEPATH "Config used by the hardware smoke test")
set(ARGUS_SENDER_TEST_PORT "55000" CACHE STRING
    "Loopback TCP port used by hardware tests")

if(ARGUS_SENDER_HARDWARE_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME avs1_hardware_smoke
        COMMAND ${Python3_EXECUTABLE}
                ${CAMERA_STREAMER_TEST_ROOT}/hardware/avs1_smoke.py
                --sender $<TARGET_FILE:argus_sender>
                --config ${ARGUS_SENDER_HARDWARE_CONFIG}
                --frames 120
                --port ${ARGUS_SENDER_TEST_PORT}
    )
    set_tests_properties(avs1_hardware_smoke PROPERTIES
        LABELS "hardware"
        RUN_SERIAL TRUE
        TIMEOUT 90
    )
endif()
