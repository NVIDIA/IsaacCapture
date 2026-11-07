# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

include(${CMAKE_SOURCE_DIR}/cmake/InstallPythonExample.cmake)
install_python_example(DESTINATION examples/realhand_ffg_glove)

# The shared Python example installer does not copy runtime environment files.
install(FILES python/isaaccapture_examples/realhand_ffg_glove/realhand_ffg_hand_demo.env
    DESTINATION examples/realhand_ffg_glove/python/isaaccapture_examples/realhand_ffg_glove
)
