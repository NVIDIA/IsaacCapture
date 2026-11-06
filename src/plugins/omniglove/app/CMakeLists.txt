# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_executable(omniglove_plugin main.cpp)

target_link_libraries(omniglove_plugin PRIVATE isaac::omniglove_plugin)

set_target_properties(omniglove_plugin PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/omniglove;$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS omniglove_plugin
    RUNTIME DESTINATION plugins/omniglove
    COMPONENT omniglove
)
