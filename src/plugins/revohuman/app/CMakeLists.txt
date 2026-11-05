# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The plugin's only source is the gloves through the C SDK, so without a library the live source
# links against there is no plugin to build.
if(NOT REVOHUMAN_LIVE_AVAILABLE)
    add_custom_target(revohuman_plugin
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping revohuman_plugin: no RevoHuman C SDK library the live source links against")
    return()
endif()

add_executable(revohuman_plugin
    main.cpp
    revohuman_plugin.cpp
    revohuman_plugin.hpp
    live_factory.cpp
)

target_link_libraries(revohuman_plugin PRIVATE
    revohuman_plugin::revohuman_live
    revohuman_plugin::revohuman_core
    Teleop::plugin_utils
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
    log_bridge::log_bridge_core
    # Device I/O owns the SDK dependency; do not obtain it through the geometry library.
    revohuman_plugin::revohuman_hand_kinematics
    revohuman_plugin::sdk
)

# The SDK library is installed next to the plugin, so the plugin looks there first.
# BUILD_RPATH adds $ORIGIN to CMake's default build-tree RPATH, which keeps the absolute SDK lib
# directory, so the build-tree binary runs in place.
set_target_properties(revohuman_plugin PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS revohuman_plugin RUNTIME DESTINATION plugins/revohuman COMPONENT revohuman)

# Every name of the shared library (librevohuman.so -> .so.<major> -> .so.<version>);
# install(FILES) keeps symlinks as symlinks. The static library is not installed.
get_filename_component(_revohuman_sdk_lib_dir "${REVOHUMAN_SDK_LIBRARY}" DIRECTORY)
file(GLOB _revohuman_sdk_shared_libs LIST_DIRECTORIES false "${_revohuman_sdk_lib_dir}/librevohuman.so*")
install(FILES ${_revohuman_sdk_shared_libs} DESTINATION plugins/revohuman COMPONENT revohuman)
# The SDK licence goes next to its library: the library is redistributed with the plugin, so a
# prefix without it (such as an old source build) is refused rather than shipped unlicensed.
if(NOT EXISTS "${REVOHUMAN_SDK_ROOT}/LICENSE")
    message(FATAL_ERROR "RevoHuman C SDK at ${REVOHUMAN_SDK_ROOT} has no LICENSE; its library cannot be "
                        "installed without it. Use a release package (install.sh downloads one).")
endif()
install(FILES "${REVOHUMAN_SDK_ROOT}/LICENSE"
    DESTINATION plugins/revohuman
    COMPONENT revohuman
    RENAME LICENSE.revohuman-sdk-c
)
