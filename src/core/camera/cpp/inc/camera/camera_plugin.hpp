// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_source.hpp"

namespace core::camera
{

/**
 * @brief Everything a camera plugin's main() does: reads the arguments, chooses the cameras,
 * takes the plugin's endpoint, opens the cameras through `backend` and serves them to readers
 * until SIGINT or SIGTERM. `--help` prints the arguments; which cameras and which endpoint
 * they give is in this library's README.md, "Running a plugin".
 *
 * Call it from main(), before another thread exists: it blocks the two signals for the
 * calling thread, so that the threads it and the backend start inherit that, and takes them
 * itself.
 *
 * @return The process's exit code: 0 after a clean stop or `--help`, 2 for a command line it
 * cannot parse, which is answered with the usage, 1 when the plugin cannot run, which is
 * logged. An argument that parses and cannot be honoured, such as an option the backend
 * refuses, returns 1.
 */
int run_camera_plugin(int argc, char** argv, const Backend& backend);

} // namespace core::camera
