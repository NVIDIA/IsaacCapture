# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(manus_glove_calibrator
  main.cpp
  manus_sdk.cpp
  manus_sdk.hpp
)

# Deliberately not manus_plugin_core: calibration is its own process with its own SDK adapter, and needs no
# OpenXR or tracking pipeline.
target_link_libraries(manus_glove_calibrator
  PRIVATE
    manus_calibration::core
    ManusSDK::ManusSDK
    log_bridge::log_bridge_core
)

set_target_properties(manus_glove_calibrator PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_WITH_INSTALL_RPATH NO
  SKIP_BUILD_RPATH NO
  BUILD_RPATH "${_ISAAC_MANUS_EFFECTIVE_BUILD_RPATH_STR}"
  INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS manus_glove_calibrator
    RUNTIME DESTINATION plugins/manus
    COMPONENT manus
)
