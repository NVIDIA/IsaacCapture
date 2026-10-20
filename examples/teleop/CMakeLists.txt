# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(teleop_examples)

include(${CMAKE_SOURCE_DIR}/cmake/InstallPythonExample.cmake)
install_python_example(DESTINATION examples/teleop/python)

# The A1 MJCF/URDF/STL package is example data, not Python package data. Keep it
# beside the installed Python examples so the demo resolves its model by path.
install(DIRECTORY assets DESTINATION examples/teleop)
