# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(jmex_plugin_tests
    test_jmex_plugin_core.cpp
    test_plugin_args.cpp
)

# The tests pass the publisher a transport of their own: no OpenXR runtime is needed.
target_link_libraries(jmex_plugin_tests PRIVATE
    Catch2::Catch2WithMain
    jmex::jmex_plugin_core
    oxr::oxr_utils
)

# The [jmex] tag becomes the ctest label the example's Python tests carry.
# The first logger opens a log file, or a forwarding socket if the shell exports one: keep the file
# in the build tree and the socket unset.
catch_discover_tests(jmex_plugin_tests
    TEST_PREFIX "jmex_"
    ADD_TAGS_AS_LABELS
    PROPERTIES ENVIRONMENT "ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs;ISAACCAPTURE_LOG_SOCKET="
)
