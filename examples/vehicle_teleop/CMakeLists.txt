# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

include(${CMAKE_SOURCE_DIR}/cmake/InstallPythonExample.cmake)
install_python_example(DESTINATION examples/vehicle_teleop/python)

install(FILES README.md
    DESTINATION examples/vehicle_teleop
)

install(DIRECTORY config
    DESTINATION examples/vehicle_teleop
)

install(PROGRAMS
    scripts/replay_command_mcap.sh
    scripts/run_isaac_keyboard_control_worker.sh
    scripts/run_isaac_remote_steering_worker.sh
    scripts/run_kia_panda_worker.sh
    DESTINATION examples/vehicle_teleop/scripts
)
