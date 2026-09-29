# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Reference examples
# ==============================================================================
# First-party code, so the project warning set applies to the whole tree. Enabled
# here rather than at the root for the reasons in src/CMakeLists.txt.
isaac_teleop_enable_compiler_warnings()

if(BUILD_EXAMPLES)
    add_subdirectory(oxr)
    add_subdirectory(retargeting)
    add_subdirectory(teleop_session_manager)
    add_subdirectory(teleop_ros2)
    add_subdirectory(schemaio)
    add_subdirectory(native_openxr)
    add_subdirectory(mcap_record_replay)
    add_subdirectory(deviceio_live_view)
    add_subdirectory(haptic_feedback)
    add_subdirectory(sharpa_avatar)
elseif(BUILD_EXAMPLE_TELEOP_ROS2)
    # Only the ROS 2 reference integration (e.g. for Docker).
    add_subdirectory(teleop_ros2)
endif()
