# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# One CTest per test file. conftest.py puts examples/orbbec_ego/python on sys.path;
# the sync fixtures compile records with the build's flatc.
file(GLOB TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py")
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name "${test_file}" NAME_WE)
    add_test(
        NAME "orbbec_ego_${test_name}"
        COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev pytest -v --tb=short "${test_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties("orbbec_ego_${test_name}" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>;EGO_TEST_FLATC=$<TARGET_FILE:flatc>"
        LABELS "orbbec_ego"
    )
endforeach()
