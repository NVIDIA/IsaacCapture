# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(teleop_examples)

include(${CMAKE_SOURCE_DIR}/cmake/InstallPythonExample.cmake)
install_python_example(DESTINATION examples/teleop/python)
