# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(ARGUS_SENDER_TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
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

add_executable(argus_sender_test_config
    tests/test_config.cpp
    src/Config.cpp
    src/CaptureBackend.cpp
)
argus_sender_add_cpp_test(argus_sender_test_config config)
target_link_libraries(argus_sender_test_config PRIVATE yaml-cpp::yaml-cpp)

add_executable(argus_sender_test_geometry_scheduler
    tests/test_geometry_scheduler.cpp
    src/OutputGeometry.cpp
    src/MixedRateScheduler.cpp
)
argus_sender_add_cpp_test(argus_sender_test_geometry_scheduler geometry_scheduler)

add_executable(argus_sender_test_nv_remap
    tests/test_nv_remap.cpp
    src/NvRemap.cpp
)
argus_sender_add_cpp_test(argus_sender_test_nv_remap nv_remap)

add_executable(argus_sender_test_rate_limiter
    tests/test_rate_limiter.cpp
    src/StereoTiming.cpp
    src/MixedRateScheduler.cpp
)
argus_sender_add_cpp_test(argus_sender_test_rate_limiter rate_limiter)

add_executable(argus_sender_test_alignment_policy
    tests/test_alignment_policy.cpp
    src/StereoAlignmentPolicy.cpp
    src/StereoTiming.cpp
)
argus_sender_add_cpp_test(argus_sender_test_alignment_policy alignment_policy)

add_executable(argus_sender_test_priority_frame_handoff
    tests/test_priority_frame_handoff.cpp
)
argus_sender_add_cpp_test(argus_sender_test_priority_frame_handoff priority_frame_handoff)

add_executable(argus_sender_test_frame_alignment_policy
    tests/test_frame_alignment_policy.cpp
    src/FrameAlignmentPolicy.cpp
)
argus_sender_add_cpp_test(argus_sender_test_frame_alignment_policy frame_alignment_policy)

add_executable(argus_sender_test_capture_domain_plan
    tests/test_capture_domain_plan.cpp
    src/CaptureDomainPlan.cpp
)
argus_sender_add_cpp_test(argus_sender_test_capture_domain_plan capture_domain_plan)

add_executable(argus_sender_test_capture_priority_policy
    tests/test_capture_priority_policy.cpp
    src/CapturePriorityPolicy.cpp
)
argus_sender_add_cpp_test(argus_sender_test_capture_priority_policy capture_priority_policy)

add_executable(argus_sender_test_avs1_wire
    tests/test_avs1_wire.cpp
    src/Avs1Wire.cpp
)
argus_sender_add_cpp_test(argus_sender_test_avs1_wire avs1_wire)

add_executable(argus_sender_test_bundle_assembler
    tests/test_bundle_assembler.cpp
    src/FrameBundleAssembler.cpp
    src/Avs1Wire.cpp
    src/StereoTiming.cpp
    src/Runtime.cpp
)
argus_sender_add_cpp_test(argus_sender_test_bundle_assembler bundle_assembler)

add_executable(argus_sender_test_tcp_transport
    tests/test_tcp_transport.cpp
    src/TcpTransport.cpp
    src/StereoTiming.cpp
    src/Runtime.cpp
)
argus_sender_add_cpp_test(argus_sender_test_tcp_transport tcp_transport)
set_tests_properties(tcp_transport PROPERTIES LABELS "loopback")

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
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/hardware/avs1_smoke.py
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

add_executable(argus_sender_test_capture_backend
    tests/test_capture_backend.cpp
    src/CaptureBackend.cpp
)
argus_sender_add_cpp_test(argus_sender_test_capture_backend capture_backend)

add_executable(argus_sender_test_encoder_input_slots
    tests/test_encoder_input_slots.cpp
)
argus_sender_add_cpp_test(argus_sender_test_encoder_input_slots encoder_input_slots)
