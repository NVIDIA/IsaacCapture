# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(schema_tests
  test_oak.cpp
  test_pose.cpp
  test_head.cpp
  test_hand.cpp
  test_full_body.cpp
  test_controller.cpp
  test_pedals.cpp
  test_steering_wheel.cpp
  test_plugin_device_status.cpp
  test_joint_se3_pose.cpp
  test_se3_tracker.cpp
  test_serialized.cpp
  test_soma_body_v0.cpp
  test_soma_body_joint_poses_v0.cpp
  test_soma_hand_v0.cpp
  test_soma_hand_joint_poses_v0.cpp
  test_schema_conform.cpp
)
target_link_libraries(schema_tests PRIVATE
  isaacteleop_schema
  schema::schema_compat
  Catch2::Catch2WithMain
  OpenXR::headers
)

# test_schema_conform.cpp reads these directories at run time. They are absolute paths into
# the tree this was built in, which is what makes the binary unfit to install.
add_dependencies(schema_tests schema_header_generation)
target_compile_definitions(schema_tests PRIVATE
  SCHEMA_FBS_DIR="${SCHEMA_FBS_DIR}"
  SCHEMA_BFBS_DIR="${SCHEMA_BFBS_DIR}"
  SCHEMA_GOLDEN_DIR="${SCHEMA_GOLDEN_DIR}"
)

# The goldens and the .fbs they are compared against are repository files, not wheel
# contents, so this binary only runs where it was built. Keep it off the venv PATH rather
# than ship one that fails on a directory belonging to the build machine.
set_target_properties(schema_tests PROPERTIES ISAACTELEOP_NEEDS_SOURCE_TREE TRUE)

message(STATUS "schema_tests target enabled with Catch2")
catch_discover_tests(schema_tests)
