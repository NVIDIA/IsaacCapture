# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Reference examples
# ==============================================================================
# First-party code, so the project warning set applies to the whole tree. Enabled
# here rather than at the root for the reasons in src/CMakeLists.txt.
isaac_teleop_enable_compiler_warnings()

add_subdirectory(oxr)
add_subdirectory(retargeting)
add_subdirectory(teleop_session_manager)
add_subdirectory(schemaio)
add_subdirectory(native_openxr)
add_subdirectory(mcap_record_replay)
add_subdirectory(deviceio_live_view)
add_subdirectory(soma_body_publisher)
add_subdirectory(soma_hand_publisher)
add_subdirectory(tactile)
add_subdirectory(haptic_feedback)
