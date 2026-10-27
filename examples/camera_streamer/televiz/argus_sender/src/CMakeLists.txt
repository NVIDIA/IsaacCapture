# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(argus_sender
    main.cpp
    Config.cpp
    Avs1Wire.cpp
    OutputGeometry.cpp
    MixedRateScheduler.cpp
    CapturePriorityPolicy.cpp
    CaptureDomainPlan.cpp
    CaptureManager.cpp
    FrameBundleAssembler.cpp
    StreamProcessor.cpp
    CaptureCoordinator.cpp
    SenderSession.cpp
    Runtime.cpp
    CameraCatalog.cpp
    FrameCopyWorker.cpp
    NvRemap.cpp
    FrameAlignmentPolicy.cpp
    StereoAlignmentPolicy.cpp
    TcpTransport.cpp
    HevcEncoder.cpp
    StereoTiming.cpp
    EGLGlobal.cpp
    NvBufSurfaceLite.cpp
    ${MMAPI_CLASS_SOURCES}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(argus_sender PRIVATE -Wall -Wextra)
    set_source_files_properties(${MMAPI_CLASS_SOURCES}
        PROPERTIES COMPILE_OPTIONS -w)
endif()

target_compile_definitions(argus_sender PRIVATE NVMMAPI_SUPPORTED)

target_include_directories(argus_sender PRIVATE
    ${ARGUS_INCLUDE_DIR}
    ${MMAPI_INCLUDE_DIR}
    ${MMAPI_CLASS_DIR}
    ${CUDA_INCLUDE_DIR}
    ${DRM_INCLUDE_DIR}
)

target_link_libraries(argus_sender PRIVATE
    ${ARGUS_LINK_LIBRARY}
    ${EGL_LIBRARY}
    ${NVV4L2_LIBRARY}
    ${NVBUFSURFACE_LIBRARY}
    ${CUDA_DRIVER_LIBRARY}
    yaml-cpp::yaml-cpp
    Threads::Threads
)

set_target_properties(argus_sender PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
install(TARGETS argus_sender DESTINATION bin)
