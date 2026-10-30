# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The real recovery backoff sleeps, and its test sets the stop flag from a second thread.
find_package(Threads REQUIRED)

add_executable(jmex_plugin_tests
    test_jmex_plugin_core.cpp
    test_plugin_args.cpp
)

# jmex_plugin_core brings in the publisher's own OpenXR transport. The tests pass a transport in
# instead, so that one is linked but never brought up, and no OpenXR runtime is needed.
target_link_libraries(jmex_plugin_tests PRIVATE
    Catch2::Catch2WithMain
    Threads::Threads
    isaac::jmex_plugin_core
    oxr::oxr_utils
)

# Each [jmex] tag becomes the ctest label the example's Python tests carry, so -L jmex runs both.
# A failed push is logged, and the first logger builds this process's own sinks: keep their file in
# the build tree, and off a forwarding socket an interactive shell may be exporting.
catch_discover_tests(jmex_plugin_tests ADD_TAGS_AS_LABELS
    PROPERTIES ENVIRONMENT "ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs;ISAACCAPTURE_LOG_SOCKET="
)
