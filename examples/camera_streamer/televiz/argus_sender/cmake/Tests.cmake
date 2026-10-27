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

add_subdirectory(tests/config)
add_subdirectory(tests/geometry_scheduler)
add_subdirectory(tests/nv_remap)
add_subdirectory(tests/rate_limiter)
add_subdirectory(tests/alignment_policy)
add_subdirectory(tests/priority_frame_handoff)
add_subdirectory(tests/frame_alignment_policy)
add_subdirectory(tests/capture_domain_plan)
add_subdirectory(tests/capture_priority_policy)
add_subdirectory(tests/avs1_wire)
add_subdirectory(tests/bundle_assembler)
add_subdirectory(tests/tcp_transport)

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
