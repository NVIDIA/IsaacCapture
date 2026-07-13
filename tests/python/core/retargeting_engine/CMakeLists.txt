# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Retargeting Engine Python Tests
# ==============================================================================

# Discover and add individual Python tests
# This makes each test show up separately in CTest output

# First, find all test files (only in this directory, not recursively)
file(GLOB TEST_FILES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py"
)

# The public robotic_grounding package is always staged with Python bindings.
# This option controls whether its heavier runtime dependencies and test run.
option(TEST_SHARPA_RETARGETER "Install grounding dependencies and run the Sharpa retargeter test" ON)

# wuji_sdk is a public PyPI package, so it installs by default and the test
# runs. OFF makes the test skip via its _HAS_WUJI_SDK guard.
option(TEST_WUJI_RETARGETER "Install the Wuji SDK so the Wuji retargeter test runs" ON)
set(WUJI_EXTRA_ARGS "")
if(TEST_WUJI_RETARGETER)
    set(WUJI_EXTRA_ARGS --extra wuji)
endif()

# Add each test file as a separate CTest test
foreach(test_file ${TEST_FILES})
    # Get test name from filename (remove .py and path)
    get_filename_component(test_name "${test_file}" NAME_WE)

    # Sharpa test alone needs the [grounding] runtime deps; other tests don't.
    set(extra_uv_args "")
    if(test_name STREQUAL "test_sharpa_hand_retargeter")
        if(NOT TEST_SHARPA_RETARGETER)
            continue()
        endif()
        set(extra_uv_args --extra grounding)
    elseif(test_name STREQUAL "test_wuji_hand_retargeter")
        set(extra_uv_args ${WUJI_EXTRA_ARGS})
    endif()

    # Add the test
    add_test(
        NAME "retargeting_${test_name}"
        COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev ${extra_uv_args} pytest -v --tb=short "${test_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # Set the environment to ensure the built package is available
    set_tests_properties("retargeting_${test_name}" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>"
    )
endforeach()

# Add mypy type checking test
add_test(
    NAME "retargeting_mypy_type_check"
    COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev
        mypy --config-file pyproject.toml
        "${CMAKE_SOURCE_DIR}/src/python/isaaccapture/retargeting_engine"
        "${CMAKE_SOURCE_DIR}/src/python/isaacteleop/retargeters/DVRK"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Set the environment to ensure the built package is available for type checking
set_tests_properties("retargeting_mypy_type_check" PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>"
)
