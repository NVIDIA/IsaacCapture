# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_test(
    NAME vehicle_teleop_test_workers
    COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev
        pytest -v --tb=short test_workers.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_tests_properties(vehicle_teleop_test_workers PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_package/$<CONFIG>"
    LABELS "vehicle_teleop"
)
