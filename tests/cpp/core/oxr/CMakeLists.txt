# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(oxr_tests test_oxr_session.cpp)
target_compile_definitions(oxr_tests PRIVATE XR_USE_TIMESPEC)
target_link_libraries(oxr_tests PRIVATE Teleop::openxr_extensions oxr::oxr_core deviceio::live_trackers deviceio::deviceio_trackers deviceio::deviceio_session Catch2::Catch2WithMain)
catch_discover_tests(oxr_tests ADD_TAGS_AS_LABELS
    PROPERTIES ENVIRONMENT "ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs;ISAACCAPTURE_LOG_SOCKET="
)

# Wrap runtime calls without loading a physical OpenXR runtime.
target_link_options(oxr_tests PRIVATE
    "LINKER:--wrap=xrCreateInstance"
    "LINKER:--wrap=xrDestroyInstance"
    "LINKER:--wrap=xrGetSystem"
    "LINKER:--wrap=xrCreateSession"
    "LINKER:--wrap=xrDestroySession"
    "LINKER:--wrap=xrCreateReferenceSpace"
    "LINKER:--wrap=xrDestroySpace"
    "LINKER:--wrap=xrEnumerateViewConfigurations"
    "LINKER:--wrap=xrBeginSession"
    "LINKER:--wrap=xrEndSession"
    "LINKER:--wrap=xrPollEvent"
    "LINKER:--wrap=xrLocateSpace"
    "LINKER:--wrap=xrGetInstanceProcAddr"
)
