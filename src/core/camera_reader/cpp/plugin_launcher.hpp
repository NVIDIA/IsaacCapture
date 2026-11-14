// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <camera_ipc/unique_fd.hpp>

#include <string>
#include <vector>

namespace core::camera
{

/**
 * @brief The camera_plugin executable a reader starts: `preferred` when given, else the one
 * next to this program, else the first on PATH.
 * @return Its path, or nothing when there is none to run.
 */
std::string find_plugin_executable(const std::string& preferred);

/**
 * @brief Start a camera plugin as a process that belongs to nobody: it outlives this process
 * and leaves no zombie behind when it exits.
 * @return A descriptor that reads end-of-file once that plugin is gone.
 * @throws std::system_error if no process could be started.
 */
UniqueFd launch_plugin(const std::string& executable, const std::vector<std::string>& arguments);

} // namespace core::camera
