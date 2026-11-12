# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# A standalone calibration process. It shares the vendor SDK's build setup with src/plugins/manus (the imported
# ManusSDK::ManusSDK target and its rpath) but none of that plugin's code.
#
# The engine and wire format need no SDK, so they build, and their tests run, without it. POSIX file and
# process calls make everything here Linux-only.
if(UNIX AND NOT APPLE)
  add_subdirectory(core)
  if(TARGET ManusSDK::ManusSDK)
    add_subdirectory(app)
  else()
    add_custom_target(manus_glove_calibrator COMMAND ${CMAKE_COMMAND} -E echo "Skipping manus_glove_calibrator: the Manus SDK was not found")
  endif()
endif()
