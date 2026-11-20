# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Robot State Python Tests (description file, records, store bindings, reader)
# ==============================================================================

file(GLOB TEST_FILES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py"
)

foreach(test_file ${TEST_FILES})
    get_filename_component(test_name "${test_file}" NAME_WE)

    add_test(
        NAME "robot_state_${test_name}"
        COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev pytest -v --tb=short "${test_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # Importing isaaccapture makes the test a logging session leader: keep its log files in the build tree.
    set_tests_properties("robot_state_${test_name}" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>;ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs"
    )
endforeach()
