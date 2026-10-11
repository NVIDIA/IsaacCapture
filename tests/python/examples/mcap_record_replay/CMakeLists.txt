# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC.
# SPDX-License-Identifier: Apache-2.0

add_test(
    NAME mcap_record_replay_ego
    COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev
        pytest -v --tb=short test_ego_replay.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(mcap_record_replay_ego PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>"
    LABELS "mcap_record_replay"
)

foreach(test_file test_ego_sync_cli test_ego_calibration_cli)
    add_test(
        NAME mcap_record_replay_${test_file}
        COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev
            pytest -v --tb=short ${test_file}.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(mcap_record_replay_${test_file} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>"
        LABELS "mcap_record_replay;ego"
    )
endforeach()
