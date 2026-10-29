# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Generates the tracker in trackers.toml, whose metadata table is fbs/, and compiles it into the
# test, so the metadata half of the tracker templates is built and run.
include(${CMAKE_SOURCE_DIR}/cmake/GenerateFlatBuffers.cmake)

generate_flatbuffer_headers(
  TEST_SCHEMA_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/fbs
  ${CMAKE_CURRENT_BINARY_DIR}/generated/schema
  TEST_SCHEMA_BFBS
)

set(_TRACKERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/trackers")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/src/core/codegen/generate_trackers.py"
          --manifest "${CMAKE_CURRENT_SOURCE_DIR}/trackers.toml"
          --defaults "${CMAKE_SOURCE_DIR}/src/core/deviceio_trackers/defaults.toml"
          --out-dir "${_TRACKERS_DIR}"
          --emit-cmake "${_TRACKERS_DIR}/generated_sources.cmake"
          --prune-stale
  RESULT_VARIABLE _tracker_gen_rc
  OUTPUT_VARIABLE _tracker_gen_out
  ERROR_VARIABLE _tracker_gen_err)
if(NOT _tracker_gen_rc EQUAL 0)
  message(FATAL_ERROR "tracker codegen failed:\n${_tracker_gen_out}\n${_tracker_gen_err}")
endif()
# The generator and templates are configure dependencies already (cmake/GenerateTrackers.cmake).
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/trackers.toml")

add_executable(tracker_metadata_tests
  test_tracker_metadata.cpp
  ${_TRACKERS_DIR}/deviceio_trackers/metadata_test_tracker.cpp
  ${_TRACKERS_DIR}/live_trackers/live_metadata_test_tracker_impl.cpp
  ${_TRACKERS_DIR}/replay_trackers/replay_metadata_test_tracker_impl.cpp
  ${TEST_SCHEMA_HEADERS}
)

# Ahead of the production generated directories, so <mcap/recording_traits.hpp> reads the
# traits of this manifest.
target_include_directories(tracker_metadata_tests BEFORE PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/generated
  ${_TRACKERS_DIR}
  ${_TRACKERS_DIR}/deviceio_trackers/inc
  ${_TRACKERS_DIR}/live_trackers
  ${_TRACKERS_DIR}/replay_trackers
  ${_TRACKERS_DIR}/inc
)

target_link_libraries(tracker_metadata_tests PRIVATE
  deviceio::live_trackers
  deviceio::replay_trackers
  # Provides the MCAP_IMPLEMENTATION symbols, as for live_trackers_tests.
  deviceio::deviceio_session
  mcap_test_support
  tensor_test_support::tensor_test_support
  Catch2::Catch2WithMain
)

message(STATUS "tracker_metadata_tests target enabled with Catch2")
catch_discover_tests(tracker_metadata_tests)
