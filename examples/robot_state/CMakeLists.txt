# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

include(${CMAKE_SOURCE_DIR}/cmake/InstallPythonExample.cmake)
install_python_example(DESTINATION examples/robot_state)

# install_python_example() takes *.py and *.md only; the stand-in robot reads these files.
install(DIRECTORY python/isaaccapture_examples/robot_state/descriptions
    DESTINATION examples/robot_state/python/isaaccapture_examples/robot_state
    FILES_MATCHING PATTERN "*.toml"
)
