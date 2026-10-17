# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The real recovery backoff sleeps, and its test sets the stop flag from a second thread.
find_package(Threads REQUIRED)

# These sources do not need the MOXI Receiver SDK, so the test compiles them directly:
# jmex_plugin_core exists only when the SDK is found. The mapping fills the generated schema's slots.
add_executable(jmex_plugin_tests
    test_jmex_plugin_core.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/jmex/core/openxr_convention.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/jmex/core/body_joint_mapping.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/jmex/core/device_clock.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/jmex/core/full_body_publisher.cpp
)

target_include_directories(jmex_plugin_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugins/jmex/core
    ${CMAKE_SOURCE_DIR}/src/plugins/jmex/core/inc
)

# oxr_core and pusherio are for the publisher's own OpenXR transport. The tests pass a transport in
# instead, so that one is linked but never brought up, and no OpenXR runtime is needed.
target_link_libraries(jmex_plugin_tests PRIVATE
    Catch2::Catch2WithMain
    Threads::Threads
    isaacteleop_schema
    oxr::oxr_core
    oxr::oxr_utils
    pusherio::pusherio
    log_bridge::log_bridge_core
)

# Each [jmex] tag becomes the ctest label the example's Python tests carry, so -L jmex runs both.
# A failed push is logged, and the first logger builds this process's own sinks: keep their file in
# the build tree, and off a forwarding socket an interactive shell may be exporting.
catch_discover_tests(jmex_plugin_tests ADD_TAGS_AS_LABELS
    PROPERTIES ENVIRONMENT "ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs;ISAACCAPTURE_LOG_SOCKET="
)
