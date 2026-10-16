# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use a writable UV_CACHE_DIR when validating from a filesystem sandbox.
# python_package is copied during build; rebuild it after changing EGO helpers.
add_test(NAME ego_embedded_recording_python
    COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev pytest -v --tb=short test_recording.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(ego_embedded_recording_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>")

foreach(ego_test sync calibration)
    add_test(NAME ego_${ego_test}_python
        COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev pytest -v --tb=short test_${ego_test}.py
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(ego_${ego_test}_python PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>;EGO_TEST_FLATC=$<TARGET_FILE:flatc>")
endforeach()
