# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
include(${CMAKE_SOURCE_DIR}/cmake/InstallPythonExample.cmake)
install_python_example(DESTINATION examples/realhand_hands)

# The shared Python example installer does not copy robot model data.
install(DIRECTORY python/isaaccapture_examples/realhand_hands/config
    DESTINATION examples/realhand_hands/python/isaaccapture_examples/realhand_hands
    FILES_MATCHING PATTERN "*.urdf" PATTERN "*.yml"
)
