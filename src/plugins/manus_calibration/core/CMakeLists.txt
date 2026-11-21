# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wire format and step engine, with no SDK dependency so they run under test without hardware.
add_library(manus_calibration_core STATIC
  engine.cpp
  export_writer.cpp
  json.cpp
  json.hpp
  protocol.cpp
  sha256.cpp
  inc/manus_calibration/engine.hpp
  inc/manus_calibration/export_writer.hpp
  inc/manus_calibration/protocol.hpp
  inc/manus_calibration/sha256.hpp
)

target_include_directories(manus_calibration_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

set_target_properties(manus_calibration_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

add_library(manus_calibration::core ALIAS manus_calibration_core)
