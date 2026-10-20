# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_test(
    NAME teleop_test_retargeting_examples
    COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} --extra dev
        pytest -v --tb=short test_retargeting_examples.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_tests_properties(teleop_test_retargeting_examples PROPERTIES LABELS "teleop")
