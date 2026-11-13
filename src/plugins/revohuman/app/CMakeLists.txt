# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(revohuman_plugin
    main.cpp
    glove_source.cpp
    glove_source.hpp
    revohuman_plugin.cpp
    revohuman_plugin.hpp
)

target_link_libraries(revohuman_plugin PRIVATE
    revohuman_plugin::revohuman_core
    revohuman_plugin::sdk
    Teleop::plugin_utils
    pusherio::pusherio
    deviceio::deviceio_session
    oxr::oxr_core
    oxr::oxr_utils
    isaacteleop_schema
    log_bridge::log_bridge_core
)

# The SDK library is installed next to the plugin. BUILD_RPATH adds $ORIGIN to the default build-tree RPATH,
# which keeps the SDK lib directory, so the build-tree binary runs in place.
set_target_properties(revohuman_plugin PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS revohuman_plugin RUNTIME DESTINATION plugins/revohuman COMPONENT revohuman)

# Every name of the shared library (.so -> .so.<major> -> .so.<version>); install(FILES) keeps symlinks.
get_filename_component(_revohuman_sdk_lib_dir "${REVOHUMAN_SDK_LIBRARY}" DIRECTORY)
file(GLOB _revohuman_sdk_shared_libs LIST_DIRECTORIES false "${_revohuman_sdk_lib_dir}/librevohuman.so*")
install(FILES ${_revohuman_sdk_shared_libs} DESTINATION plugins/revohuman COMPONENT revohuman)
install(FILES "${REVOHUMAN_SDK_ROOT}/LICENSE"
    DESTINATION plugins/revohuman
    COMPONENT revohuman
    RENAME LICENSE.revohuman-sdk-c
)
